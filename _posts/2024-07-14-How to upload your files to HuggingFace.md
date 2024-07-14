---
title: 'How to upload your files to HuggingFace'
date: 2024-07-14
permalink: /posts/2024/07/How to upload your files to HuggingFace/
tags:
  - HuggingFace
---

`huggingface_hub` library helps you interact with the Hub without leaving your development environment. You can create and manage repositories easily, download and upload files, and get useful model and dataset metadata from the Hub.

# Step 1: Installation

To get started, install the huggingface_hub library:

```bash
pip install --upgrade huggingface_hub
```

For more details, check out the [installation](https://huggingface.co/docs/huggingface_hub/v0.23.4/en/installation) guide.

# Step 2: Authentication

In a lot of cases, you must be authenticated with a Hugging Face account to interact with the Hub: download private repos, upload files, create PRs,... Create an account if you don’t already have one, and then sign in to get your **User Access Token** from your **Settings** page. The User Access Token is used to authenticate your identity to the Hub.

> :bulb:Tokens can have **read** or **write** permissions. Make sure to have a write access token if you want to create or edit a repository. Otherwise, it’s best to generate a read token to reduce risk in case your token is inadvertently leaked.

# Step 3: Login

The easiest way to authenticate is to save the token on your machine. You can do that from the terminal using the login() command:

```bash
huggingface-cli login
```

The command will tell you if you are already logged in and prompt you for your token. The token is then validated and saved in your `HF_HOME` directory (defaults to `~/.cache/huggingface/token`). Any script or library interacting with the Hub will use this token when sending requests.

# Step 4: Upload Files

The full command for uploading files is here:

```bash
huggingface-cli <command> [<args>] upload [-h] [--repo-type {model,dataset,space}] [--revision REVISION]
                                          [--private] [--include [INCLUDE ...]] [--exclude [EXCLUDE ...]]
                                          [--delete [DELETE ...]] [--commit-message COMMIT_MESSAGE]
                                          [--commit-description COMMIT_DESCRIPTION] [--create-pr]
                                          [--every EVERY] [--token TOKEN] [--quiet]
                                          repo_id [local_path] [path_in_repo]
```

you should specify the necessary keywords like:

```bash
huggingface-cli upload --repo-type model Wauplin/my-cool-model ./models/model.safetensors model.safetensors
```


or

```bash
huggingface-cli upload Wauplin/my-cool-model ./models .
```

---

For more details, click [here](https://huggingface.co/docs/huggingface_hub/v0.23.4/en/quick-start#upload-files).