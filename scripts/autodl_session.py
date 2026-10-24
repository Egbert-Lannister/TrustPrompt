"""Operator-driven SSH/SFTP session; password is read once, never stored.

The remote host key must already be recorded by an interactive SSH connection.
No SSH key is installed, no remote authentication settings are changed.
JSON lines on stdin: exec(command), put(local, remote), get(remote, local), close.
"""
import argparse
import getpass
import json
from pathlib import Path
import time
import paramiko


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--user", default="root")
    p.add_argument("--known-hosts", type=Path, required=True)
    a = p.parse_args()
    client = paramiko.SSHClient()
    client.load_host_keys(str(a.known_hosts))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    password = getpass.getpass("SSH password (memory only): ")
    client.connect(a.host, port=a.port, username=a.user, password=password,
                   look_for_keys=False, allow_agent=False, timeout=30, auth_timeout=30)
    del password
    client.get_transport().set_keepalive(30)
    print(json.dumps({"status": "connected", "host": a.host, "port": a.port}), flush=True)
    try:
        while True:
            try:
                line = input()
            except EOFError:
                break
            if not line.strip():
                continue
            try:
                request = json.loads(line)
                op = request["op"]
                if op == "close":
                    break
                if op in {"put", "get"}:
                    with client.open_sftp() as sftp:
                        if op == "put":
                            sftp.put(request["local"], request["remote"])
                        else:
                            Path(request["local"]).parent.mkdir(parents=True, exist_ok=True)
                            sftp.get(request["remote"], request["local"])
                    result = {"status": "ok", "op": op}
                elif op == "exec":
                    channel = client.get_transport().open_session()
                    channel.set_combine_stderr(True)
                    channel.exec_command(request["command"])
                    chunks = bytearray()
                    started = time.monotonic()
                    while not channel.exit_status_ready() or channel.recv_ready():
                        if channel.recv_ready():
                            chunks.extend(channel.recv(65536))
                            if len(chunks) > 262144:
                                del chunks[:-262144]
                        else:
                            time.sleep(.1)
                        if time.monotonic() - started > request.get("timeout", 50):
                            channel.close()
                            raise TimeoutError("Use detached jobs for long remote commands")
                    result = {"status": "ok", "exit_code": channel.recv_exit_status(),
                              "output": chunks.decode("utf-8", errors="replace")}
                    channel.close()
                else:
                    raise ValueError("unknown operation")
                print(json.dumps(result, ensure_ascii=True), flush=True)
            except Exception as e:
                print(json.dumps({"status": "error", "error": str(e)}, ensure_ascii=True), flush=True)
    finally:
        client.close()


if __name__ == "__main__":
    main()
