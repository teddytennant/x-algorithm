use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow, bail};
use arc_swap::ArcSwapOption;
use log::{info, warn};
use tokio::process::Command;

use crate::metrics::{
    CONFIG_SYNC_APPLIED, CONFIG_SYNC_FAILURES, CONFIG_SYNC_HEAD_AGE_SECS,
    CONFIG_SYNC_LAST_SUCCESS_SECS, CONFIG_SYNC_REVISION,
};

pub const FEATURES_FILE: &str = "features/home-mixer/main/rust_home_mixer.yml";
pub const ABDECIDER_FILE: &str = "abdecider/abdecider.yml";

#[derive(Clone, Debug)]
pub struct ConfigSyncOptions {
    pub remote: String,
    pub branch: String,
    pub root: PathBuf,
    pub interval: Duration,
    pub initial_sync_timeout: Duration,
}

pub struct ConfigSync {
    options: ConfigSyncOptions,
    revision: ArcSwapOption<String>,
}

impl ConfigSync {
    pub fn live_dir(root: &Path) -> PathBuf {
        root.join("live")
    }

    pub fn features_path(root: &Path) -> PathBuf {
        Self::live_dir(root).join("rust_home_mixer.yml")
    }

    pub fn abdecider_path(root: &Path) -> PathBuf {
        Self::live_dir(root).join("abdecider.yml")
    }

    pub async fn start(options: ConfigSyncOptions) -> Result<Arc<Self>> {
        let sync = Arc::new(Self {
            options,
            revision: ArcSwapOption::empty(),
        });
        sync.initial_sync().await?;
        Ok(sync)
    }

    pub fn revision(&self) -> Option<String> {
        self.revision.load().as_deref().cloned()
    }

    pub fn spawn_poller(self: &Arc<Self>) {
        let sync = Arc::clone(self);
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(sync.options.interval);
            ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            ticker.tick().await;
            loop {
                ticker.tick().await;
                if let Err(e) = sync.sync_once().await {
                    CONFIG_SYNC_FAILURES.with_label_values(&["poll"]).inc();
                    warn!(
                        "config sync failed, keeping revision {:?}: {e:#}",
                        sync.revision()
                    );
                }
            }
        });
    }

    async fn initial_sync(&self) -> Result<()> {
        let deadline = tokio::time::Instant::now() + self.options.initial_sync_timeout;
        let mut backoff = Duration::from_secs(2);
        loop {
            match self.sync_once().await {
                Ok(_) => return Ok(()),
                Err(e) => {
                    CONFIG_SYNC_FAILURES.with_label_values(&["initial"]).inc();
                    if tokio::time::Instant::now() + backoff > deadline {
                        return Err(e.context("initial config sync did not succeed in time"));
                    }
                    warn!("initial config sync failed, retrying in {backoff:?}: {e:#}");
                    tokio::time::sleep(backoff).await;
                    backoff = (backoff * 2).min(Duration::from_secs(30));
                }
            }
        }
    }

    async fn sync_once(&self) -> Result<bool> {
        let repo = self.options.root.join("repo");
        if !repo.join(".git").is_dir() {
            self.clone_sparse(&repo).await?;
        }
        git(
            &repo,
            &[
                "fetch",
                "-q",
                "--depth",
                "1",
                "origin",
                &self.options.branch,
            ],
        )
        .await?;
        let fetched = git(&repo, &["rev-parse", "FETCH_HEAD"])
            .await?
            .trim()
            .to_string();
        let current = self.revision();
        if current.as_deref() == Some(fetched.as_str()) {
            self.record_success(&repo, &fetched).await;
            return Ok(false);
        }
        git(&repo, &["reset", "-q", "--hard", "FETCH_HEAD"]).await?;
        self.publish(&repo)?;
        self.revision.store(Some(Arc::new(fetched.clone())));
        CONFIG_SYNC_APPLIED.inc();
        CONFIG_SYNC_REVISION.reset();
        CONFIG_SYNC_REVISION.with_label_values(&[&fetched]).set(1);
        self.record_success(&repo, &fetched).await;
        info!(
            "config sync applied revision {fetched} (previous {})",
            current.as_deref().unwrap_or("none")
        );
        Ok(true)
    }

    async fn clone_sparse(&self, repo: &Path) -> Result<()> {
        if repo.exists() {
            std::fs::remove_dir_all(repo)
                .with_context(|| format!("removing incomplete checkout {}", repo.display()))?;
        }
        std::fs::create_dir_all(&self.options.root)
            .with_context(|| format!("creating {}", self.options.root.display()))?;
        let repo_str = repo
            .to_str()
            .ok_or_else(|| anyhow!("non-UTF-8 config root {}", repo.display()))?;
        git(
            &self.options.root,
            &[
                "clone",
                "-q",
                "--depth",
                "1",
                "--filter=blob:none",
                "--sparse",
                "--no-checkout",
                "--branch",
                &self.options.branch,
                &self.options.remote,
                repo_str,
            ],
        )
        .await?;
        git(
            repo,
            &[
                "sparse-checkout",
                "set",
                "--no-cone",
                FEATURES_FILE,
                ABDECIDER_FILE,
            ],
        )
        .await?;
        Ok(())
    }

    fn publish(&self, repo: &Path) -> Result<()> {
        let live = Self::live_dir(&self.options.root);
        std::fs::create_dir_all(&live).with_context(|| format!("creating {}", live.display()))?;
        for (source, target) in [
            (
                repo.join(FEATURES_FILE),
                Self::features_path(&self.options.root),
            ),
            (
                repo.join(ABDECIDER_FILE),
                Self::abdecider_path(&self.options.root),
            ),
        ] {
            let tmp = target.with_extension("yml.tmp");
            std::fs::copy(&source, &tmp)
                .with_context(|| format!("copying {} to {}", source.display(), tmp.display()))?;
            std::fs::rename(&tmp, &target)
                .with_context(|| format!("renaming {} to {}", tmp.display(), target.display()))?;
        }
        Ok(())
    }

    async fn record_success(&self, repo: &Path, revision: &str) {
        CONFIG_SYNC_LAST_SUCCESS_SECS.set(unix_secs_now());
        if let Ok(out) = git(repo, &["show", "-s", "--format=%ct", revision]).await
            && let Ok(committed) = out.trim().parse::<f64>()
        {
            CONFIG_SYNC_HEAD_AGE_SECS.set(unix_secs_now() - committed);
        }
    }
}

async fn git(cwd: &Path, args: &[&str]) -> Result<String> {
    let output = Command::new("git")
        .args(args)
        .current_dir(cwd)
        .env("GIT_TERMINAL_PROMPT", "0")
        .stdin(Stdio::null())
        .output()
        .await
        .with_context(|| format!("spawning git {}", args.join(" ")))?;
    if !output.status.success() {
        bail!(
            "git {} failed ({}): {}",
            args.join(" "),
            output.status,
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}

fn unix_secs_now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}
