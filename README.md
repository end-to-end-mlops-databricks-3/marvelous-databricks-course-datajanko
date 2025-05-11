<h1 align="center">
Marvelous MLOps End-to-end MLOps with Databricks course

## Use Case
We're going to use data of Counterstrike: Global Offensive matches from [kaggle](https://www.kaggle.com/datasets/mateusdmachado/csgo-professional-matches/data). There are a lot of use cases that we could treat with this dataset. For example:
- Predicting the probability of a team winning a map before the map starts
- Predicting while a map takes places, who will win the map.
- We could do regression on how many rounds a team will win on the T (Terrorist) or CT (Counter Terrorist) side
- Recommending which maps to pick for a match against a specific team
- We could do regression for player statistics

Since we rather want to focus on MLOps and allow for fast iterations, we'll consider to predict the winning probability of the individual teams for a given map. The raw results dataset mainly contains categorical features, however, since we are having event based data, we can provide some lagged features, like overall winning percentage per last $n$ maps over all maps, winning percentages only for a given map, etc.

This also implies that we need a temporal split of the training data and we could consider appropriate cross validation schemes similar to the ones used in time-series forecasting.


## Practical information
- Weekly lectures on Wednesdays 16:00-18:00 CET.
- Code for the lecture is shared before the lecture.
- Presentation and lecture materials are shared right after the lecture.
- Video of the lecture is uploaded within 24 hours after the lecture.

- Every week we set up a deliverable, and you implement it with your own dataset.
- To submit the deliverable, create a feature branch in that repository, and a PR to main branch. The code can be merged after we review & approve & CI pipeline runs successfully.
- The deliverables can be submitted with a delay (for example, lecture 1 & 2 together), but we expect you to finish all assignments for the course before the 15th of June.


## Set up your environment
In this course, we use Databricks 15.4 LTS runtime, which uses Python 3.11.
In our examples, we use UV. Check out the documentation on how to install it: https://docs.astral.sh/uv/getting-started/installation/

Install task: https://taskfile.dev/installation/

Update .env file with the following:
```
GIT_TOKEN=<your github PAT>
```

To create a new environment and create a lockfile, run:
```
task sync-dev
source .venv/bin/activate
```

Or, alternatively:
```
export GIT_TOKEN=<your github PAT>
uv venv -p 3.11 .venv
source .venv/bin/activate
uv sync --extra dev
```
