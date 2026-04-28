"""Eval question suite.

Each `Question` carries:
  * `name`: stable id for results aggregation (qNN_<slug> mirrors pipe_NN
    in tinybirdco/llm-benchmark so results can be lined up side-by-side).
  * `prompt`: natural-language question to send to the LLM. The text is
    copied (and lightly de-typo'd) from the upstream `.pipe` files.
  * `reference_sql`: a per-dialect dict of reference SQL. The dialect key
    matches `SqlExecutor.dialect_label()` substring matched (so 'sqlite'
    and 'sqlite (in-memory)' both pick up the 'sqlite' entry).

The full 50-question suite from `tinybirdco/llm-benchmark` is ported to
SQLite-flavored SQL that runs against the synthetic `github_events` seed
in `eval/dataset/github_events.py`. ClickHouse-only constructs are
translated as follows:

  uniq()/uniqExact()           -> count(DISTINCT ...)
  uniqIf(col, cond)            -> count(DISTINCT CASE WHEN cond THEN col END)
  arrayJoin(labels)            -> recursive CTE that splits comma-separated text
  toYear / toDate / toDayOfWeek-> strftime('%Y' / %Y-%m-%d / '%w', ...)
  match(ref, '/(main|master)$')-> ref LIKE '%/main' OR ref LIKE '%/master'
  ILIKE                        -> LIKE        (SQLite LIKE is ASCII case-insensitive)
  exp10/log10                  -> pow(10, log10(...))   (math ext is enabled in 3.35+)
  position(s, c)               -> instr(s, c)
  substring(s, 1, n)           -> substr(s, 1, n)
  concat(a, b)                 -> a || b
  LIMIT 1 BY repo              -> ROW_NUMBER() OVER (PARTITION BY repo ORDER BY ...)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Question:
    name: str
    prompt: str
    reference_sql: dict[str, str]
    notes: str = ""
    # Optional per-question database pointer. Set by question sources
    # whose schema is per-question rather than global -- e.g. BIRD-SQL
    # where each question targets a different ``.sqlite`` file. The
    # runner threads this through to ``SqlExecutor.execute(sql,
    # question=...)`` so executors that care (e.g. BirdSqliteExecutor)
    # can swap their connection per question. Executors with a single
    # global schema (sqlite/synthetic/tinybird) ignore it.
    db_path: str | None = None


# Each entry is `(name, prompt, sqlite_reference_sql)`. Keeping them as a
# tuple list (rather than a giant Python literal of `Question(...)` calls)
# makes diffs reviewable and keeps the file under 1k lines.
_RAW: list[tuple[str, str, str]] = [
    (
        "q01_count_stars",
        "Count all stars (WatchEvent rows).",
        "SELECT count(*) AS stars FROM github_events WHERE event_type = 'WatchEvent'",
    ),
    (
        "q02_top_starred_repos",
        "Top 10 repositories by stars.",
        """
        SELECT repo_name, count(*) AS stars
        FROM github_events
        WHERE event_type = 'WatchEvent'
        GROUP BY repo_name
        ORDER BY stars DESC, repo_name ASC
        LIMIT 10
        """,
    ),
    (
        "q03_stars_powers_of_10",
        "For each power of 10 (1, 10, 100, 1000, etc.), count how many "
        "GitHub repositories have approximately that many stars. Order the "
        "results from smallest to largest star count.",
        """
        WITH per_repo AS (
            SELECT repo_name, count(*) AS c
            FROM github_events
            WHERE event_type = 'WatchEvent'
            GROUP BY repo_name
        )
        SELECT pow(10.0, floor(log10(c))) AS stars,
               count(DISTINCT repo_name)  AS repos
        FROM per_repo
        GROUP BY stars
        ORDER BY stars ASC
        """,
    ),
    (
        "q04_total_repos",
        "The total number of repositories on GitHub (in this dataset).",
        "SELECT count(DISTINCT repo_name) AS repos FROM github_events",
    ),
    (
        "q05_top_repos_by_year_since_2015",
        "How has the list of top repositories changed over the years from "
        "2015 onwards? Return year, repo (lowercased), and the star count.",
        """
        SELECT
            CAST(strftime('%Y', created_at) AS INTEGER) AS year,
            lower(repo_name) AS repo,
            count(*) AS c
        FROM github_events
        WHERE event_type = 'WatchEvent'
          AND CAST(strftime('%Y', created_at) AS INTEGER) >= 2015
        GROUP BY year, repo
        ORDER BY year ASC, c DESC, repo ASC
        LIMIT 10
        """,
    ),
    (
        "q06_stars_by_year",
        "How has the total number of stars changed over time, by year?",
        """
        SELECT CAST(strftime('%Y', created_at) AS INTEGER) AS year,
               count(*) AS stars
        FROM github_events
        WHERE event_type = 'WatchEvent'
        GROUP BY year
        ORDER BY year ASC
        """,
    ),
    (
        "q07_top_actors_giving_stars",
        "Who are the top 10 people giving stars?",
        """
        SELECT actor_login, count(*) AS stars
        FROM github_events
        WHERE event_type = 'WatchEvent'
        GROUP BY actor_login
        ORDER BY stars DESC, actor_login ASC
        LIMIT 10
        """,
    ),
    (
        "q08_repos_starred_by_tensorflow_starrers",
        "What are the top 10 repositories sorted by the number of stars "
        "from people who starred the tensorflow/tensorflow repository? "
        "Exclude tensorflow/tensorflow itself.",
        """
        SELECT repo_name, count(*) AS stars
        FROM github_events
        WHERE event_type = 'WatchEvent'
          AND actor_login IN (
              SELECT actor_login FROM github_events
              WHERE event_type = 'WatchEvent'
                AND repo_name = 'tensorflow/tensorflow'
          )
          AND repo_name != 'tensorflow/tensorflow'
        GROUP BY repo_name
        ORDER BY stars DESC, repo_name ASC
        LIMIT 10
        """,
    ),
    (
        "q09_top_users_alice_repo_ratio",
        "Find the top 10 GitHub users who have the highest proportion of "
        "their stars given to repositories that user 'alice' has starred, "
        "compared to all other repositories they've starred. Show each "
        "user's name, the number of stars they gave to alice's repos, the "
        "number of stars they gave to other repos, and the ratio of "
        "alice-repo stars to total stars (rounded to 3 decimals).",
        """
        WITH alice_repos AS (
            SELECT DISTINCT repo_name FROM github_events
            WHERE event_type = 'WatchEvent' AND actor_login = 'alice'
        )
        SELECT
            actor_login,
            sum(CASE WHEN repo_name IN (SELECT repo_name FROM alice_repos)
                     THEN 1 ELSE 0 END) AS stars_my,
            sum(CASE WHEN repo_name NOT IN (SELECT repo_name FROM alice_repos)
                     THEN 1 ELSE 0 END) AS stars_other,
            round(
                1.0 * sum(CASE WHEN repo_name IN (SELECT repo_name FROM alice_repos)
                               THEN 1 ELSE 0 END) / count(*),
                3
            ) AS ratio
        FROM github_events
        WHERE event_type = 'WatchEvent'
        GROUP BY actor_login
        ORDER BY ratio DESC, actor_login ASC
        LIMIT 10
        """,
    ),
    (
        "q10_non_tensorflow_repos_with_tf_contributors",
        "Find the top 10 non-TensorFlow repositories that received the "
        "most pull requests from TensorFlow contributors, ranked by the "
        "number of unique contributors.",
        """
        SELECT repo_name, count(DISTINCT actor_login) AS authors
        FROM github_events
        WHERE event_type = 'PullRequestEvent' AND action = 'opened'
          AND actor_login IN (
              SELECT actor_login FROM github_events
              WHERE event_type = 'PullRequestEvent' AND action = 'opened'
                AND repo_name LIKE '%tensorflow%'
          )
          AND repo_name NOT LIKE '%tensorflow%'
        GROUP BY repo_name
        ORDER BY authors DESC, repo_name ASC
        LIMIT 10
        """,
    ),
    (
        "q11_repos_with_issues_from_tf_issue_authors",
        "Top 10 repositories with the most issues from users who also "
        "opened issues in tensorflow/tensorflow, ranked by author count.",
        """
        SELECT repo_name,
               count(*) AS prs,
               count(DISTINCT actor_login) AS authors
        FROM github_events
        WHERE event_type = 'IssuesEvent' AND action = 'opened'
          AND actor_login IN (
              SELECT actor_login FROM github_events
              WHERE event_type = 'IssuesEvent' AND action = 'opened'
                AND repo_name = 'tensorflow/tensorflow'
          )
          AND repo_name NOT LIKE '%tensorflow%'
        GROUP BY repo_name
        ORDER BY authors DESC, repo_name ASC
        LIMIT 10
        """,
    ),
    (
        "q12_repos_top_single_day_stars",
        "Top 10 repositories with the most stars in any single day. Limit "
        "to one row per repository (the best day) and 10 rows total.",
        """
        WITH per_repo_day AS (
            SELECT
                repo_name,
                date(created_at) AS day,
                count(*) AS stars,
                ROW_NUMBER() OVER (
                    PARTITION BY repo_name
                    ORDER BY count(*) DESC, date(created_at) ASC
                ) AS rn
            FROM github_events
            WHERE event_type = 'WatchEvent'
            GROUP BY repo_name, day
        )
        SELECT repo_name, day, stars
        FROM per_repo_day
        WHERE rn = 1
        ORDER BY stars DESC, repo_name ASC
        LIMIT 10
        """,
    ),
    (
        "q13_yoy_growth_2017_vs_2016",
        "Top 10 repositories with the highest year-over-year growth "
        "between 2016 and 2017, starting from at least 1 star in 2016. "
        "Return repo_name, stars2017, stars2016, and yoy percentage "
        "(rounded to 2 decimals).",
        """
        WITH per_repo_year AS (
            SELECT repo_name,
                   CAST(strftime('%Y', created_at) AS INTEGER) AS year
            FROM github_events
            WHERE event_type = 'WatchEvent'
        )
        SELECT
            repo_name,
            sum(CASE WHEN year = 2017 THEN 1 ELSE 0 END) AS stars2017,
            sum(CASE WHEN year = 2016 THEN 1 ELSE 0 END) AS stars2016,
            round(
                100.0 * (
                    sum(CASE WHEN year = 2017 THEN 1 ELSE 0 END)
                    - sum(CASE WHEN year = 2016 THEN 1 ELSE 0 END)
                ) / sum(CASE WHEN year = 2016 THEN 1 ELSE 0 END),
                2
            ) AS yoy
        FROM per_repo_year
        GROUP BY repo_name
        HAVING stars2016 >= 1
        ORDER BY yoy DESC, repo_name ASC
        LIMIT 10
        """,
    ),
    (
        "q14_yoy_stagnation_2017_vs_2016",
        "Top 10 repositories with the worst stagnation (lowest year-over-"
        "year change) in 2017 vs 2016. Only include repos with at least "
        "one star in 2017.",
        """
        WITH per_repo_year AS (
            SELECT repo_name,
                   CAST(strftime('%Y', created_at) AS INTEGER) AS year
            FROM github_events
            WHERE event_type = 'WatchEvent'
        )
        SELECT
            repo_name,
            sum(CASE WHEN year = 2017 THEN 1 ELSE 0 END) AS stars2017,
            sum(CASE WHEN year = 2016 THEN 1 ELSE 0 END) AS stars2016,
            round(
                100.0 * (
                    sum(CASE WHEN year = 2017 THEN 1 ELSE 0 END)
                    - sum(CASE WHEN year = 2016 THEN 1 ELSE 0 END)
                ) / sum(CASE WHEN year = 2016 THEN 1 ELSE 0 END),
                2
            ) AS yoy
        FROM per_repo_year
        GROUP BY repo_name
        HAVING stars2017 >= 1
        ORDER BY yoy ASC, repo_name ASC
        LIMIT 10
        """,
    ),
    (
        "q15_steady_star_growth",
        "Top 10 repositories with the most steady star growth rate over "
        "time. Return repo_name, daily_stars (the busiest day), "
        "total_stars, and the rate (total / max-daily).",
        """
        WITH per_repo_day AS (
            SELECT repo_name, date(created_at) AS day, count(*) AS stars
            FROM github_events
            WHERE event_type = 'WatchEvent'
            GROUP BY repo_name, day
        )
        SELECT repo_name,
               max(stars) AS daily_stars,
               sum(stars) AS total_stars,
               1.0 * sum(stars) / max(stars) AS rate
        FROM per_repo_day
        GROUP BY repo_name
        ORDER BY rate DESC, repo_name ASC
        LIMIT 10
        """,
    ),
    (
        "q16_best_day_of_week_for_stars",
        "What is the best day of the week to catch a star? Return day-of-"
        "week and star count, ordered by day-of-week.",
        """
        SELECT CAST(strftime('%w', created_at) AS INTEGER) AS day,
               count(*) AS stars
        FROM github_events
        WHERE event_type = 'WatchEvent'
        GROUP BY day
        ORDER BY day ASC
        """,
    ),
    (
        "q17_total_users",
        "The total number of users on GitHub (distinct actor_login).",
        "SELECT count(DISTINCT actor_login) AS users FROM github_events",
    ),
    (
        "q18_users_with_stars",
        "Total number of users that gave at least one star.",
        "SELECT count(DISTINCT actor_login) AS users FROM github_events "
        "WHERE event_type = 'WatchEvent'",
    ),
    (
        "q19_users_with_pushes",
        "Total number of users with at least one push.",
        "SELECT count(DISTINCT actor_login) AS users FROM github_events "
        "WHERE event_type = 'PushEvent'",
    ),
    (
        "q20_users_with_opened_pr",
        "Total number of users with at least one created (opened) PR.",
        "SELECT count(DISTINCT actor_login) AS users FROM github_events "
        "WHERE event_type = 'PullRequestEvent' AND action = 'opened'",
    ),
    (
        "q21_top_starred_among_pr_users",
        "Top 10 starred repositories, but only counting stars from users "
        "who have opened at least one PR.",
        """
        SELECT repo_name, count(*) AS c
        FROM github_events
        WHERE event_type = 'WatchEvent'
          AND actor_login IN (
              SELECT actor_login FROM github_events
              WHERE event_type = 'PullRequestEvent' AND action = 'opened'
          )
        GROUP BY repo_name
        ORDER BY c DESC, repo_name ASC
        LIMIT 10
        """,
    ),
    (
        "q22_top_starred_among_10pr_users",
        "Top 10 repositories starred by users who have made at least 10 "
        "opened PRs across the dataset.",
        """
        SELECT repo_name, count(*) AS c
        FROM github_events
        WHERE event_type = 'WatchEvent'
          AND actor_login IN (
              SELECT actor_login FROM github_events
              WHERE event_type = 'PullRequestEvent' AND action = 'opened'
              GROUP BY actor_login
              HAVING count(*) >= 10
          )
        GROUP BY repo_name
        ORDER BY c DESC, repo_name ASC
        LIMIT 10
        """,
    ),
    (
        "q23_top_repos_by_opened_prs",
        "Top 10 repositories with the maximum number of opened pull requests.",
        """
        SELECT repo_name, count(*) AS c
        FROM github_events
        WHERE event_type = 'PullRequestEvent' AND action = 'opened'
        GROUP BY repo_name
        ORDER BY c DESC, repo_name ASC
        LIMIT 10
        """,
    ),
    (
        "q24_top_repos_by_pr_contributors",
        "Top 10 repositories with the most pull-request contributors "
        "(distinct authors who opened a PR).",
        """
        SELECT repo_name, count(DISTINCT actor_login) AS u
        FROM github_events
        WHERE event_type = 'PullRequestEvent' AND action = 'opened'
        GROUP BY repo_name
        ORDER BY u DESC, repo_name ASC
        LIMIT 10
        """,
    ),
    (
        "q25_top_repos_by_issues",
        "Top 10 repositories with the maximum number of opened issues.",
        """
        SELECT repo_name, count(*) AS c
        FROM github_events
        WHERE event_type = 'IssuesEvent' AND action = 'opened'
        GROUP BY repo_name
        ORDER BY c DESC, repo_name ASC
        LIMIT 10
        """,
    ),
    (
        "q26_top_repos_by_pushers",
        "Top 10 repositories with the most people who have push access "
        "(distinct PushEvent authors).",
        """
        SELECT
            repo_name,
            count(DISTINCT CASE WHEN event_type = 'PushEvent'
                                THEN actor_login END) AS u
        FROM github_events
        WHERE event_type IN ('PushEvent', 'WatchEvent')
        GROUP BY repo_name
        ORDER BY u DESC, repo_name ASC
        LIMIT 10
        """,
    ),
    (
        "q27_top_repos_by_main_branch_pushers",
        "Top 10 repositories with the most people who pushed to the main "
        "or master branch (ref ends in /main or /master).",
        """
        SELECT
            repo_name,
            count(DISTINCT CASE
                WHEN event_type = 'PushEvent'
                 AND (ref LIKE '%/main' OR ref LIKE '%/master')
                THEN actor_login END) AS u
        FROM github_events
        WHERE event_type IN ('PushEvent', 'WatchEvent')
        GROUP BY repo_name
        ORDER BY u DESC, repo_name ASC
        LIMIT 10
        """,
    ),
    (
        "q28_top_repos_main_pushers_with_100_stars",
        "Top 10 repositories with the most people who pushed to main/master, "
        "but only repositories with at least 100 stars (and repo_name != '/').",
        """
        SELECT
            repo_name,
            count(DISTINCT CASE
                WHEN event_type = 'PushEvent'
                 AND (ref LIKE '%/main' OR ref LIKE '%/master')
                THEN actor_login END) AS u,
            sum(CASE WHEN event_type = 'WatchEvent' THEN 1 ELSE 0 END) AS stars
        FROM github_events
        WHERE event_type IN ('PushEvent', 'WatchEvent')
          AND repo_name != '/'
        GROUP BY repo_name
        HAVING stars >= 100
        ORDER BY u DESC, repo_name ASC
        LIMIT 10
        """,
    ),
    (
        "q29_top_repos_by_member_invitations",
        "Top 10 repositories by member invitations "
        "(MemberEvent rows where action = 'added').",
        """
        SELECT repo_name, count(*) AS invitations
        FROM github_events
        WHERE event_type = 'MemberEvent' AND action = 'added'
        GROUP BY repo_name
        ORDER BY invitations DESC, repo_name ASC
        LIMIT 10
        """,
    ),
    (
        "q30_most_forked_repos",
        "Top 10 most forked repositories.",
        """
        SELECT repo_name, count(*) AS forks
        FROM github_events
        WHERE event_type = 'ForkEvent'
        GROUP BY repo_name
        ORDER BY forks DESC, repo_name ASC
        LIMIT 10
        """,
    ),
    (
        "q31_top_repos_by_stars_forks_ratio",
        "Top 10 repositories by the ratio between stars and forks. "
        "Return repo_name, forks, stars, and the rounded ratio "
        "(stars/forks, 3 decimals). Order by forks descending.",
        """
        SELECT
            repo_name,
            sum(CASE WHEN event_type = 'ForkEvent' THEN 1 ELSE 0 END) AS forks,
            sum(CASE WHEN event_type = 'WatchEvent' THEN 1 ELSE 0 END) AS stars,
            round(
                1.0 * sum(CASE WHEN event_type = 'WatchEvent' THEN 1 ELSE 0 END)
                    / sum(CASE WHEN event_type = 'ForkEvent' THEN 1 ELSE 0 END),
                3
            ) AS ratio
        FROM github_events
        WHERE event_type IN ('ForkEvent', 'WatchEvent')
        GROUP BY repo_name
        ORDER BY forks DESC, repo_name ASC
        LIMIT 10
        """,
    ),
    (
        "q32_total_stars_forks_ratio",
        "Total number of stars, total number of forks, and the ratio "
        "(rounded to 2 decimals) between stars and forks.",
        """
        SELECT
            sum(CASE WHEN event_type = 'WatchEvent' THEN 1 ELSE 0 END) AS stars,
            sum(CASE WHEN event_type = 'ForkEvent' THEN 1 ELSE 0 END)  AS forks,
            round(
                1.0 * sum(CASE WHEN event_type = 'WatchEvent' THEN 1 ELSE 0 END)
                    / sum(CASE WHEN event_type = 'ForkEvent' THEN 1 ELSE 0 END),
                2
            ) AS ratio
        FROM github_events
        WHERE event_type IN ('WatchEvent', 'ForkEvent')
        """,
    ),
    (
        "q33_total_opened_issues",
        "Total number of issues opened on GitHub.",
        "SELECT count(*) AS issues FROM github_events "
        "WHERE event_type = 'IssuesEvent' AND action = 'opened'",
    ),
    (
        "q34_top_repos_by_issue_comments",
        "Top 10 repositories by issue-comment events.",
        """
        SELECT repo_name, count(*) AS c
        FROM github_events
        WHERE event_type = 'IssueCommentEvent'
        GROUP BY repo_name
        ORDER BY c DESC, repo_name ASC
        LIMIT 10
        """,
    ),
    (
        "q35_top_repos_by_comments_per_issue",
        "Top 10 repositories by ratio between issue comments and issues. "
        "Return repo_name, comments, distinct issue numbers, and the "
        "rounded ratio (comments / issues, 2 decimals). Order by comments "
        "descending.",
        """
        SELECT repo_name,
               count(*) AS comments,
               count(DISTINCT number) AS issues,
               round(1.0 * count(*) / count(DISTINCT number), 2) AS ratio
        FROM github_events
        WHERE event_type = 'IssueCommentEvent'
        GROUP BY repo_name
        ORDER BY comments DESC, repo_name ASC
        LIMIT 10
        """,
    ),
    (
        "q36_top_issues_by_comment_count",
        "Top 10 GitHub issues (repo + issue number) by how many issue "
        "comments have been created on them.",
        """
        SELECT repo_name, number, count(*) AS comments
        FROM github_events
        WHERE event_type = 'IssueCommentEvent' AND action = 'created'
        GROUP BY repo_name, number
        ORDER BY comments DESC, repo_name ASC, number ASC
        LIMIT 10
        """,
    ),
    (
        "q37_top_issues_by_comments_with_10_authors",
        "Top 10 GitHub issues with number > 10 by how many comments have "
        "been created. Also return the number of distinct comment authors "
        "and require at least 10 of them.",
        """
        SELECT repo_name,
               number,
               count(*) AS comments,
               count(DISTINCT actor_login) AS authors
        FROM github_events
        WHERE event_type = 'IssueCommentEvent'
          AND action = 'created'
          AND number > 10
        GROUP BY repo_name, number
        HAVING authors >= 10
        ORDER BY comments DESC, repo_name ASC, number ASC
        LIMIT 10
        """,
    ),
    (
        "q38_repos_with_tensorflow_mentions",
        "Top 10 repositories with the most events whose body mentions "
        "tensorflow. Return repo_name and the number of matching events.",
        """
        SELECT repo_name, count(*) AS c
        FROM github_events
        WHERE body LIKE '%tensorflow%'
        GROUP BY repo_name
        ORDER BY c DESC, repo_name ASC
        LIMIT 10
        """,
    ),
    (
        "q39_top_repos_by_commit_comments",
        "Top 10 repositories by the number of commit comments.",
        """
        SELECT repo_name, count(*) AS comments
        FROM github_events
        WHERE event_type = 'CommitCommentEvent'
        GROUP BY repo_name
        ORDER BY comments DESC, repo_name ASC
        LIMIT 10
        """,
    ),
    (
        "q40_top_commits_by_comment_count",
        "Top 10 commits by the count of commit-comment events. Only include "
        "commits with at least 10 distinct commenters. Return the full "
        "commit URL ('https://github.com/{repo}/commit/{commit_id}'), "
        "the total comments, and the number of unique authors.",
        """
        SELECT
            'https://github.com/' || repo_name || '/commit/' || commit_id AS URL,
            count(*) AS comments,
            count(DISTINCT actor_login) AS authors
        FROM github_events
        WHERE event_type = 'CommitCommentEvent'
          AND commit_id != ''
        GROUP BY repo_name, commit_id
        HAVING authors >= 10
        ORDER BY comments DESC, URL ASC
        LIMIT 10
        """,
    ),
    (
        "q41_toughest_code_reviews",
        "Top 10 most tough code reviews (PRs with the most distinct "
        "comment authors). Return the full URL of the PR "
        "('https://github.com/{repo}/pull/{number}') and the number of "
        "unique authors.",
        """
        SELECT
            'https://github.com/' || repo_name || '/pull/' || CAST(number AS TEXT) AS URL,
            count(DISTINCT actor_login) AS authors
        FROM github_events
        WHERE event_type = 'PullRequestReviewCommentEvent'
          AND action = 'created'
        GROUP BY repo_name, number
        ORDER BY authors DESC, URL ASC
        LIMIT 10
        """,
    ),
    (
        "q42_top_authors_by_pushes",
        "Top 10 authors with the most pushes.",
        """
        SELECT actor_login, count(*) AS c
        FROM github_events
        WHERE event_type = 'PushEvent'
        GROUP BY actor_login
        ORDER BY c DESC, actor_login ASC
        LIMIT 10
        """,
    ),
    (
        "q43_top_orgs_by_stars",
        "Top 10 users/organizations by the number of stars (derive the "
        "organization name from the prefix of repo_name up to and "
        "including the slash, lowercased).",
        """
        SELECT
            lower(substr(repo_name, 1, instr(repo_name, '/'))) AS org,
            count(*) AS stars
        FROM github_events
        WHERE event_type = 'WatchEvent'
        GROUP BY org
        ORDER BY stars DESC, org ASC
        LIMIT 10
        """,
    ),
    (
        "q44_top_orgs_by_repos",
        "Top 10 organizations by the number of (popular) repositories. "
        "Take repos with at least 10 stars; group by lowercased org "
        "prefix; return org and distinct repo count.",
        """
        SELECT
            lower(substr(repo_name, 1, instr(repo_name, '/'))) AS org,
            count(DISTINCT repo_name) AS repos
        FROM (
            SELECT repo_name
            FROM github_events
            WHERE event_type = 'WatchEvent'
            GROUP BY repo_name
            HAVING count(*) >= 10
        )
        GROUP BY org
        ORDER BY repos DESC, org ASC
        LIMIT 10
        """,
    ),
    (
        "q45_top_repos_by_pr_churn",
        "Top 10 repositories ranked by total code churn (additions + "
        "deletions) from PRs that were opened. Only include PRs with "
        "additions < 10000 and deletions < 10000. Return repo_name, PR "
        "count, unique authors, total adds, and total dels.",
        """
        SELECT repo_name,
               count(*) AS prs,
               count(DISTINCT actor_login) AS authors,
               sum(additions) AS adds,
               sum(deletions) AS dels
        FROM github_events
        WHERE event_type = 'PullRequestEvent'
          AND action = 'opened'
          AND additions < 10000
          AND deletions < 10000
        GROUP BY repo_name
        ORDER BY (sum(additions) + sum(deletions)) DESC, repo_name ASC
        LIMIT 10
        """,
    ),
    (
        "q46_top_authors_by_code_reviews",
        "Top 10 authors with the most code reviews. Return actor_login, "
        "the number of pull-request review comments created, the number "
        "of distinct repos, and the number of distinct PRs they "
        "commented on.",
        """
        SELECT
            actor_login,
            count(*) AS comments,
            count(DISTINCT repo_name) AS repos,
            count(DISTINCT repo_name || '#' || CAST(number AS TEXT)) AS prs
        FROM github_events
        WHERE event_type = 'PullRequestReviewCommentEvent'
          AND action = 'created'
        GROUP BY actor_login
        ORDER BY comments DESC, actor_login ASC
        LIMIT 10
        """,
    ),
    (
        "q47_top_labels",
        "Top 10 most popular labels for issues and pull requests. Labels "
        "are stored as a comma-separated string. Return the label and the "
        "number of times it has been used in IssuesEvent / "
        "PullRequestEvent / IssueCommentEvent rows whose action is in "
        "('created', 'opened', 'labeled').",
        """
        WITH RECURSIVE
        splits(label, rest) AS (
            SELECT '', labels || ','
            FROM github_events
            WHERE event_type IN ('IssuesEvent', 'PullRequestEvent', 'IssueCommentEvent')
              AND action IN ('created', 'opened', 'labeled')
              AND labels != ''
            UNION ALL
            SELECT substr(rest, 1, instr(rest, ',') - 1),
                   substr(rest, instr(rest, ',') + 1)
            FROM splits
            WHERE rest != '' AND instr(rest, ',') > 0
        )
        SELECT label, count(*) AS c
        FROM splits
        WHERE label != ''
        GROUP BY label
        ORDER BY c DESC, label ASC
        LIMIT 10
        """,
    ),
    (
        "q48_top_orgs_by_community",
        "Top 10 organizations by community size. Return the lowercased "
        "org prefix; total distinct authors; and the per-event-type "
        "distinct author counts: PR authors, issue authors, issue-comment "
        "authors, PR-review-comment authors, and push authors. Only "
        "consider rows whose event_type is one of those five.",
        """
        SELECT
            lower(substr(repo_name, 1, instr(repo_name, '/'))) AS org,
            count(DISTINCT actor_login) AS authors,
            count(DISTINCT CASE WHEN event_type = 'PullRequestEvent'
                                THEN actor_login END) AS pr_authors,
            count(DISTINCT CASE WHEN event_type = 'IssuesEvent'
                                THEN actor_login END) AS issue_authors,
            count(DISTINCT CASE WHEN event_type = 'IssueCommentEvent'
                                THEN actor_login END) AS comment_authors,
            count(DISTINCT CASE WHEN event_type = 'PullRequestReviewCommentEvent'
                                THEN actor_login END) AS review_authors,
            count(DISTINCT CASE WHEN event_type = 'PushEvent'
                                THEN actor_login END) AS push_authors
        FROM github_events
        WHERE event_type IN (
            'PullRequestEvent',
            'IssuesEvent',
            'IssueCommentEvent',
            'PullRequestReviewCommentEvent',
            'PushEvent'
        )
        GROUP BY org
        ORDER BY authors DESC, org ASC
        LIMIT 10
        """,
    ),
    (
        "q49_longest_repo_names",
        "Top 10 longest repository names with at least 1 star. Return "
        "repo_name and its length.",
        """
        SELECT repo_name, length(repo_name) AS name_length
        FROM github_events
        WHERE event_type = 'WatchEvent'
        GROUP BY repo_name
        ORDER BY name_length DESC, repo_name ASC
        LIMIT 10
        """,
    ),
    (
        "q50_shortest_repo_names",
        "Top 10 shortest repository names with at least 1 star, in "
        "owner/repo form. Return only the repository name.",
        """
        SELECT repo_name
        FROM github_events
        WHERE event_type = 'WatchEvent'
          AND repo_name LIKE '%_/_%'
        GROUP BY repo_name
        ORDER BY length(repo_name) ASC, repo_name ASC
        LIMIT 10
        """,
    ),
]


def _normalize_sql(sql: str) -> str:
    """Strip and collapse internal blank lines for cleaner JSON output."""
    return "\n".join(line.rstrip() for line in sql.strip().splitlines())


QUESTIONS: list[Question] = [
    Question(
        name=name,
        prompt=prompt,
        reference_sql={"sqlite": _normalize_sql(sql)},
    )
    for name, prompt, sql in _RAW
]


def select(
    names: list[str] | None = None,
    *,
    limit: int | None = None,
) -> list[Question]:
    """Return all questions, optionally filtered by name and/or capped.

    Args:
        names: optional list of question names. If provided, only those
               questions are returned (in the requested order). Unknown
               names raise ``KeyError``.
        limit: optional maximum number of questions to return. Applied
               *after* the name filter. ``None`` (default) means no cap.
    """
    if names is None:
        out = list(QUESTIONS)
    else:
        by_name = {q.name: q for q in QUESTIONS}
        missing = [n for n in names if n not in by_name]
        if missing:
            raise KeyError(f"Unknown question(s): {missing}")
        out = [by_name[n] for n in names]
    if limit is not None and limit >= 0:
        out = out[:limit]
    return out


__all__ = ["QUESTIONS", "Question", "select"]
