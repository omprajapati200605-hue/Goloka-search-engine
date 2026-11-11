# Deploying Goloka (website + API in one container)

The site and the API run as ONE app, so you deploy one thing. Anyone can search; only someone with
your secret key can add pages (`GOLOKA_PUBLIC_READ=true` + `GOLOKA_API_KEY`).

## Option A - Hugging Face Spaces (free, no credit card)

1. Create the Space: huggingface.co -> New -> Space -> SDK **Docker** -> name `goloka` -> Public.
2. Put this block at the very TOP of `README.md` (above the existing `# ...` title), keep the `---` lines:

       ---
       title: Goloka
       sdk: docker
       app_port: 7860
       ---

3. Push the project to the Space (cmd, from the project folder):

       git remote add space https://huggingface.co/spaces/YOUR_USERNAME/goloka
       git push space main

   (When asked for a password, use a Hugging Face access token with *write* permission.)
4. Space -> Settings -> **Variables and secrets** -> New secret:
   - `GOLOKA_API_KEY` = a long random password you choose (this is your "add pages" key)
   - New variable `GOLOKA_USER_AGENT` = `GOLOKABot/1.0 (+https://huggingface.co/spaces/YOUR_USERNAME/goloka)`
5. Wait for the build (about 5-10 minutes). Open `https://YOUR_USERNAME-goloka.hf.space`.

Free Spaces have no permanent disk: pages you add are lost when the Space restarts or sleeps.
Re-add them with Add pages, or use a paid persistent-storage add-on (then set the data paths to `/data`).

## Option B - Fly.io (keeps data permanently, needs a card, about $3-6 per month)

    fly launch --no-deploy --internal-port 7860
    fly volumes create goloka_data --size 1
    fly secrets set GOLOKA_API_KEY=your-long-password
    fly deploy

In `fly.toml` add:  `[mounts]  source = "goloka_data"  destination = "/data"`
and give the machine at least 1 GB RAM (`fly scale memory 1024`).

## Settings the container uses
| Variable | Value | Meaning |
|---|---|---|
| GOLOKA_API_HOST | 0.0.0.0 | listen for outside traffic |
| GOLOKA_API_PORT | 7860 | port (change if your host wants another) |
| GOLOKA_PUBLIC_READ | true | anyone can search |
| GOLOKA_API_KEY | (secret, you set it) | needed to add pages |
| GOLOKA_DATA_DIR / DB_PATH / VECTOR_DIR | /data... | where data is stored |

Never put the API key in a file in the repository. Set it only as a secret in the host's settings.
