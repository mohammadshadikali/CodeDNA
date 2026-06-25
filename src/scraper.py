import argparse
import csv
import os
import sys
import tempfile

from git import Repo, GitCommandError


def open_repository(repo_path_or_url, clone_dir=None):
    if os.path.isdir(repo_path_or_url) and os.path.isdir(os.path.join(repo_path_or_url, ".git")):
        return Repo(repo_path_or_url)

    clone_dir = clone_dir or tempfile.mkdtemp(prefix="git_repo_clone_")
    try:
        return Repo.clone_from(repo_path_or_url, clone_dir)
    except GitCommandError as exc:
        raise RuntimeError(f"Unable to open or clone repository '{repo_path_or_url}': {exc}")


def extract_commits(repo):
    for commit in repo.iter_commits(rev="HEAD", reverse=True):
        files_changed = sorted(commit.stats.files.keys())
        yield {
            "commit_hash": commit.hexsha,
            "author_name": commit.author.name,
            "author_email": commit.author.email,
            "date": commit.authored_datetime.isoformat(),
            "message": commit.message.strip(),
            "files_changed": ";".join(files_changed),
        }


def write_commits_csv(commits, output_path):
    fieldnames = [
        "commit_hash",
        "author_name",
        "author_email",
        "date",
        "message",
        "files_changed",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for commit_data in commits:
            writer.writerow(commit_data)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clone or read a local git repository and export commits to commits.csv"
    )
    parser.add_argument(
        "repository",
        help="Local repository path or remote Git repository URL",
    )
    parser.add_argument(
        "--output",
        default="commits.csv",
        help="CSV output file path (default: commits.csv)",
    )
    parser.add_argument(
        "--clone-dir",
        help="Directory to clone repository into when a remote URL is provided",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        repo = open_repository(args.repository, clone_dir=args.clone_dir)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    commits = extract_commits(repo)
    write_commits_csv(commits, args.output)
    print(f"Wrote commit history to {args.output}")


if __name__ == "__main__":
    main()
