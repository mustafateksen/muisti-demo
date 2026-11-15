# MUISTI demo

A browser demo of [MUISTI](https://github.com/mustafateksen/MUISTI): a phone mock-up that plays recorded scenes, and a question box that asks the live models.

- `index.html`: the page (no build step).
- `data/demo.js`: the demo memory and the answers recorded from the app.
- `api/ask.py`: the live question (embed, rerank, Nemotron answer over `data/server.json`). Runs as a Vercel Python function; needs the environment variable `VERDA_KEY`. Set `MUISTI_LIVE=off` to switch live answers off.
- `config.js`: where the page finds the live server.
- `img/CREDITS.md`: picture credits.

MIT licence, see the main repository.
