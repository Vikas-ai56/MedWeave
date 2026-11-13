# Running MedWeave in Docker

The whole system in containers on one machine: the coordinator, 4 institutions and the dashboard,
talking to each other over TLS exactly as they would on separate laptops.

## Quick start

```bash
docker compose up -d --build            # build the image, then setup, coordinator, institutions, dashboard
docker compose run --rm session         # run a training session and stream its log
open http://127.0.0.1:8080              # dashboard
docker compose down                     # stop everything (data, checkpoints and certificates are kept)
```

- **The first build** downloads the Python and Node packages. **The first `up`** downloads PathMNIST
  (205 MB) into the `data` volume. Later starts reuse both.
- **The first session** spends about 25 s on Flower's one-time `uv sync` before the first turn.
- **Run settings** go in `SESSION_CONFIG`, in the same format as `run_local.sh --run-config`:

  ```bash
  SESSION_CONFIG="num-rounds=1 batches-per-turn=20 noise-std=0.25" docker compose run --rm session
  ```

- **A new session continues** from the latest Backbone and each institution's saved Head and Tail.
  To start from v0: `docker compose down -v` (this also deletes the data and certificates, so the next
  `up` re-downloads and re-creates them).

## What runs

| Service | Command | Volumes |
|---|---|---|
| `setup` (runs once, then exits) | `make_certs.sh /certs coordinator localhost`, then `python -m institution_data prepare` | `certs`, `data` |
| `coordinator` | `flower-superlink --host 0.0.0.0` with the server certificate. It runs the ServerApp and the HTTPS relay on port 8765 | `certs` (read-only), `coordinator-state` |
| `inst-1` … `inst-4` | `flower-supernode --root-certificates /certs/ca.crt --superlink coordinator:8000 --node-config 'institution-id="inst-N"'` | `certs`, `data` (read-only), `inst-N-state` |
| `dashboard` | `uvicorn dashboard.backend.app:app` on `127.0.0.1:8080` | `coordinator-state` |
| `session` (on demand) | `flwr run coordinator/flower_app docker --stream` with `relay-host="0.0.0.0"` and `relay-url="https://coordinator:8765"` | `certs` (read-only) |

- Every process uses the same image (`medweave:latest`): Python 3.14, CPU-only PyTorch, the system
  packages and the built dashboard. It runs as an unprivileged user.
- Each institution has its own state volume, so Heads, Tails and optimizer state stay separate, as on
  real machines.
- **One difference from real deployments:** the `data` volume is shared read-only. Each institution still
  opens only its own shard (the manifest check refuses any other), but on separate laptops each machine
  would hold only its own shard.

## TLS

`setup` creates a private CA and the coordinator's certificate (valid for `coordinator`, `localhost` and
`127.0.0.1`) in the `certs` volume:
- The SuperLink and the relay use the certificate.
- The SuperNodes, the `flwr` CLI and the institutions' relay clients verify it with `ca.crt`.
- Nothing runs with `--insecure`.

## Troubleshooting

| Problem | Fix |
|---|---|
| `docker compose run --rm session` waits at "registering" | An institution is down: `docker compose ps`, then `docker compose logs inst-N` |
| Dashboard shows "database not ready" | No session has started yet; it fills in at the first registration |
| `port is already allocated` for 8080 | Another dashboard is running: stop `dashboard/run_dashboard.sh`, or change the port mapping |
| Changed code is not picked up | `docker compose up -d --build` rebuilds the image |
| Start again from Backbone v0 | `docker compose down -v` |
