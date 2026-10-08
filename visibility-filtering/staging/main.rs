#![deny(clippy::allow_attributes, clippy::allow_attributes_without_reason)]

use clap::{Parser, Subcommand};
use std::path::PathBuf;
use xai_visibility_filtering_service::server::{self, ServeArgs};
use xai_visibility_filtering_service::staging::capture;
use xai_visibility_filtering_service::staging::serve::StagingServer;

#[derive(Parser, Debug)]
#[command(about = "Visibility Filtering staging tools")]
struct Args {
    #[arg(long, default_value = "atla")]
    datacenter: String,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    Serve {
        #[command(flatten)]
        serve: ServeArgs,
        #[arg(long)]
        test_users: bool,
    },
    Capture {
        #[arg(long)]
        test_users: bool,
        #[arg(long, default_value = capture::FIXTURES_IN_IMAGE)]
        fixtures: PathBuf,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    match args.command {
        Command::Serve { serve, test_users } => {
            server::serve::<StagingServer>(serve, test_users.then(capture::test_users_metadata))
                .await
        }
        Command::Capture {
            test_users,
            fixtures,
        } => capture::run(&args.datacenter, test_users, &fixtures).await,
    }
}
