#!/usr/bin/env bash
# Create the TLS files for a MedWeave deployment with openssl:
#   ca.crt, ca.key          a private certificate authority (keep ca.key on the coordinator only)
#   server.crt, server.key  the coordinator's certificate, used by the Flower SuperLink and the relay
#
#   ./make_certs.sh <out-dir> [extra-name ...]
#
# The server certificate is valid for localhost and 127.0.0.1 plus every extra name: the coordinator
# laptop's LAN IP (e.g. 192.168.1.10) or a hostname (e.g. coordinator, for Docker). Institutions get
# only ca.crt. Existing files are kept, so re-running is safe; delete the directory to start over.
set -euo pipefail
OUT=${1:?usage: make_certs.sh <out-dir> [extra-name ...]}
shift
mkdir -p "$OUT"
cd "$OUT"
if [[ -f server.crt && -f server.key && -f ca.crt ]]; then
  echo "TLS files already in $OUT"
  exit 0
fi

SAN="DNS:localhost,IP:127.0.0.1"
for name in "$@"; do
  case $name in
    *:* | *[0-9].*[0-9]) case $name in *[!0-9a-fA-F.:]*) SAN="$SAN,DNS:$name" ;; *) SAN="$SAN,IP:$name" ;; esac ;;
    *) SAN="$SAN,DNS:$name" ;;
  esac
done

openssl req -x509 -newkey rsa:4096 -sha256 -days 365 -nodes \
  -keyout ca.key -out ca.crt -subj "/CN=MedWeave CA" \
  -addext "basicConstraints=critical,CA:TRUE" -addext "keyUsage=critical,keyCertSign,cRLSign" \
  -addext "subjectKeyIdentifier=hash" 2>/dev/null
openssl req -newkey rsa:4096 -sha256 -nodes -keyout server.key -out server.csr -subj "/CN=MedWeave coordinator" 2>/dev/null
openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key -CAcreateserial -days 365 -sha256 -out server.crt \
  -extfile <(printf "subjectAltName=%s\nbasicConstraints=CA:FALSE\nkeyUsage=critical,digitalSignature,keyEncipherment\nextendedKeyUsage=serverAuth\nsubjectKeyIdentifier=hash\nauthorityKeyIdentifier=keyid,issuer\n" "$SAN") 2>/dev/null
rm -f server.csr ca.srl
chmod 600 ca.key server.key
echo "TLS files written to $OUT (server certificate valid for $SAN)"
