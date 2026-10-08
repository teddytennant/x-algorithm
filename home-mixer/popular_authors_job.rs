use std::collections::HashMap;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::clients::popular_authors_store_client::ManhattanPopularAuthorsStore;
use crate::util::popular_authors::{
    now_ms, select_top_posting_authors, PopularAuthor, PopularAuthorsStore, StoredPopularAuthors,
};

const METRIC_PREFIX: &str = "PopularAuthorsJob";
const BIGQUERY_API: &str = "https://bigquery.googleapis.com/bigquery/v2";
const BIGQUERY_SCOPE: &str = "https://www.googleapis.com/auth/bigquery.readonly";
const PAGE_SIZE: usize = 10_000;
const HTTP_TIMEOUT: Duration = Duration::from_secs(60);

pub struct JobConfig {
    pub datacenter: String,
    pub interval: Duration,
    pub fraction: f64,
    pub bigquery_table: String,
    pub bigquery_key_path: String,
    pub egress_proxy: Option<String>,
    pub max_snapshot_age: Duration,
    pub once: bool,
}

#[derive(Debug)]
pub struct Snapshot {
    pub generated_at_ms: i64,
    pub active_posters_7d: u64,
    pub by_followers_desc: Vec<PopularAuthor>,
}

#[derive(Deserialize)]
struct ServiceAccountKey {
    client_email: String,
    private_key: String,
    token_uri: String,
}

#[derive(Serialize)]
struct JwtClaims<'a> {
    iss: &'a str,
    scope: &'a str,
    aud: &'a str,
    iat: i64,
    exp: i64,
}

#[derive(Deserialize)]
struct TokenResponse {
    access_token: String,
}

#[derive(Deserialize)]
struct TableMetadata {
    schema: TableSchema,
}

#[derive(Deserialize)]
struct TableSchema {
    fields: Vec<TableField>,
}

#[derive(Deserialize)]
struct TableField {
    name: String,
}

#[derive(Deserialize)]
struct TableData {
    #[serde(rename = "pageToken")]
    page_token: Option<String>,
    rows: Option<Vec<TableRow>>,
}

#[derive(Deserialize)]
struct TableRow {
    f: Vec<TableCell>,
}

#[derive(Deserialize)]
struct TableCell {
    v: Option<String>,
}

struct BigQueryTable {
    http: reqwest::Client,
    key: ServiceAccountKey,
    url: String,
}

fn gauge(name: &str, value: f64) {
    if let Some(receiver) = xai_stats_receiver::global_stats_receiver() {
        receiver.gauge(&format!("{METRIC_PREFIX}.{name}"), &[], value);
    }
}

fn count_run(result: &str) {
    if let Some(receiver) = xai_stats_receiver::global_stats_receiver() {
        receiver.incr(&format!("{METRIC_PREFIX}.Runs"), &[("result", result)], 1);
    }
}

impl BigQueryTable {
    fn new(config: &JobConfig) -> anyhow::Result<Self> {
        let key: ServiceAccountKey =
            serde_json::from_str(&std::fs::read_to_string(&config.bigquery_key_path)?)?;
        let mut http = reqwest::Client::builder().timeout(HTTP_TIMEOUT);
        if let Some(proxy) = &config.egress_proxy {
            http = http.proxy(reqwest::Proxy::all(proxy)?);
        }
        let parts: Vec<&str> = config.bigquery_table.split('.').collect();
        anyhow::ensure!(
            parts.len() == 3,
            "bigquery table must be project.dataset.table, got {}",
            config.bigquery_table
        );
        Ok(Self {
            http: http.build()?,
            key,
            url: format!(
                "{BIGQUERY_API}/projects/{}/datasets/{}/tables/{}",
                parts[0], parts[1], parts[2]
            ),
        })
    }

    async fn access_token(&self) -> Result<String, String> {
        let now = now_ms() / 1000;
        let claims = JwtClaims {
            iss: &self.key.client_email,
            scope: BIGQUERY_SCOPE,
            aud: &self.key.token_uri,
            iat: now - 60,
            exp: now + 3000,
        };
        let signing_key = jsonwebtoken::EncodingKey::from_rsa_pem(self.key.private_key.as_bytes())
            .map_err(|e| format!("bigquery key: {e}"))?;
        let assertion = jsonwebtoken::encode(
            &jsonwebtoken::Header::new(jsonwebtoken::Algorithm::RS256),
            &claims,
            &signing_key,
        )
        .map_err(|e| format!("bigquery jwt: {e}"))?;
        let response: TokenResponse = self
            .http
            .post(&self.key.token_uri)
            .form(&[
                ("grant_type", "urn:ietf:params:oauth:grant-type:jwt-bearer"),
                ("assertion", assertion.as_str()),
            ])
            .send()
            .await
            .and_then(reqwest::Response::error_for_status)
            .map_err(|e| format!("bigquery token: {e}"))?
            .json()
            .await
            .map_err(|e| format!("bigquery token response: {e}"))?;
        Ok(response.access_token)
    }

    async fn get<T: serde::de::DeserializeOwned>(
        &self,
        token: &str,
        url: &str,
    ) -> Result<T, String> {
        self.http
            .get(url)
            .bearer_auth(token)
            .send()
            .await
            .and_then(reqwest::Response::error_for_status)
            .map_err(|e| format!("bigquery GET {url}: {e}"))?
            .json()
            .await
            .map_err(|e| format!("bigquery response {url}: {e}"))
    }

    async fn read_rows(&self) -> Result<Vec<HashMap<String, String>>, String> {
        let token = self.access_token().await?;
        let metadata: TableMetadata = self.get(&token, &self.url).await?;
        let names: Vec<String> = metadata.schema.fields.into_iter().map(|f| f.name).collect();
        let mut rows = Vec::new();
        let mut page_token: Option<String> = None;
        loop {
            let mut url = format!("{}/data?maxResults={PAGE_SIZE}", self.url);
            if let Some(t) = &page_token {
                url.push_str(&format!("&pageToken={t}"));
            }
            let page: TableData = self.get(&token, &url).await?;
            for row in page.rows.unwrap_or_default() {
                rows.push(
                    names
                        .iter()
                        .cloned()
                        .zip(row.f.into_iter().map(|c| c.v.unwrap_or_default()))
                        .collect(),
                );
            }
            match page.page_token {
                Some(t) => page_token = Some(t),
                None => return Ok(rows),
            }
        }
    }
}

fn field<T: std::str::FromStr>(row: &HashMap<String, String>, name: &str) -> Result<T, String> {
    row.get(name)
        .ok_or_else(|| format!("snapshot row has no {name}"))?
        .parse()
        .map_err(|_| format!("snapshot {name} is not a number: {:?}", row.get(name)))
}

pub fn parse_snapshot(rows: &[HashMap<String, String>]) -> Result<Snapshot, String> {
    let first = rows
        .first()
        .ok_or("popular posting authors snapshot is empty")?;
    let generated_at_secs: f64 = field(first, "generated_at")?;
    let active_posters_7d: u64 = field(first, "active_posters_7d")?;
    let mut ranked = Vec::with_capacity(rows.len());
    for row in rows {
        ranked.push((
            field::<u64>(row, "follower_rank")?,
            PopularAuthor {
                author_id: field(row, "author_id")?,
                follower_count: field(row, "follower_count")?,
            },
        ));
    }
    ranked.sort_by_key(|(rank, _)| *rank);
    Ok(Snapshot {
        generated_at_ms: (generated_at_secs * 1000.0) as i64,
        active_posters_7d,
        by_followers_desc: ranked.into_iter().map(|(_, a)| a).collect(),
    })
}

struct Job {
    table: BigQueryTable,
    store: ManhattanPopularAuthorsStore,
    fraction: f64,
    max_snapshot_age: Duration,
}

impl Job {
    async fn run_once(&self) -> Result<(), String> {
        let snapshot = parse_snapshot(&self.table.read_rows().await?)?;
        let age_ms = now_ms() - snapshot.generated_at_ms;
        gauge("SnapshotAgeHours", age_ms as f64 / 3_600_000.0);
        if age_ms > self.max_snapshot_age.as_millis() as i64 {
            return Err(format!(
                "popular posting authors snapshot is {:.1} h old",
                age_ms as f64 / 3_600_000.0
            ));
        }
        let authors = select_top_posting_authors(
            &snapshot.by_followers_desc,
            snapshot.active_posters_7d,
            self.fraction,
        )?;
        let min_followers = authors.last().map_or(0, |a| a.follower_count);
        let stored = StoredPopularAuthors {
            updated_at_ms: snapshot.generated_at_ms,
            authors,
        };
        let current = self.store.load().await?;
        if current.as_ref() != Some(&stored) {
            self.store.save(&stored).await?;
            let read_back = self.store.load().await?;
            if read_back.as_ref() != Some(&stored) {
                return Err("popular authors read back does not match what was written".to_string());
            }
            tracing::info!(
                authors = stored.authors.len(),
                "popular authors list updated in manhattan"
            );
        }
        tracing::info!(
            percent = self.fraction * 100.0,
            active_posters_7d = snapshot.active_posters_7d,
            authors = stored.authors.len(),
            min_followers,
            snapshot_age_hours = age_ms as f64 / 3_600_000.0,
            "picked the top {}% of authors who posted in the last 7 days, by follower count",
            self.fraction * 100.0
        );
        gauge("LastSuccessUnixSecs", now_ms() as f64 / 1000.0);
        gauge("Authors", stored.authors.len() as f64);
        gauge("ActivePosters7d", snapshot.active_posters_7d as f64);
        gauge("MinFollowerCount", min_followers as f64);
        gauge("Fraction", self.fraction);
        Ok(())
    }
}

pub async fn run(config: JobConfig) -> anyhow::Result<()> {
    let job = Job {
        table: BigQueryTable::new(&config)?,
        store: ManhattanPopularAuthorsStore::new(&config.datacenter).await?,
        fraction: config.fraction,
        max_snapshot_age: config.max_snapshot_age,
    };
    tracing::info!(
        fraction = config.fraction,
        table = %config.bigquery_table,
        interval_secs = config.interval.as_secs(),
        "popular authors job started"
    );
    loop {
        let started = Instant::now();
        match job.run_once().await {
            Ok(()) => count_run("ok"),
            Err(e) => {
                count_run("error");
                tracing::error!(error = %e, "popular authors job run failed");
                if config.once {
                    anyhow::bail!(e);
                }
            }
        }
        if config.once {
            return Ok(());
        }
        tokio::time::sleep(config.interval.saturating_sub(started.elapsed())).await;
    }
}
