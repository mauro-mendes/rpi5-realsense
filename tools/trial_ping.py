"""
trial_ping.py — teste LEVE do trial-sync: valida ZMQ (porta 5571) + NTP/chrony
                SEM subir câmera/Hailo. Roda em segundos.

Uso:
  # na BENGALA (escravo), sobe um REP mínimo:
  python tools/trial_ping.py --serve

  # no rpi5-realsense (mestre), pinga a bengala:
  python tools/trial_ping.py --ping 192.168.1.162

O --ping manda PING, mede o RTT e estima o offset de relógio (wall_remoto − wall_local).
Se o offset for de poucos ms → chrony sincronizado. Se não responder → porta 5571
bloqueada / servidor fora do ar / IP errado.

ATENÇÃO: não rode --serve junto com `main.py --trials` (ambos usam a porta 5571).
Este script é só o pré-teste; o teste real é o main.py --trials + record_trajectory.py.
"""

import argparse
import json
import time

import zmq


def serve(port: int) -> None:
    ctx = zmq.Context.instance()
    s = ctx.socket(zmq.REP)
    s.bind(f"tcp://*:{port}")
    print(f"[serve] REP em tcp://*:{port} — Ctrl+C p/ sair", flush=True)
    try:
        while True:
            raw = s.recv()
            wall = time.time()
            try:
                cmd = json.loads(raw.decode())
            except Exception:
                cmd = {}
            print(f"[serve] recebido: {cmd}", flush=True)
            s.send(json.dumps({"ack": cmd.get("cmd", "?"), "wall": wall}).encode())
    except KeyboardInterrupt:
        print("\n[serve] encerrado.")


def ping(ip: str, port: int, n: int) -> None:
    ctx = zmq.Context.instance()
    s = ctx.socket(zmq.REQ)
    s.setsockopt(zmq.RCVTIMEO, 2500)
    s.setsockopt(zmq.LINGER, 0)
    s.connect(f"tcp://{ip}:{port}")
    print(f"[ping] → tcp://{ip}:{port}", flush=True)
    offs = []
    for i in range(n):
        t0 = time.time()
        s.send(json.dumps({"cmd": "PING", "wall": t0}).encode())
        try:
            rep = json.loads(s.recv().decode())
        except Exception as e:
            print(f"[ping] SEM RESPOSTA ({e}) — porta {port} bloqueada? servidor no ar? IP certo?")
            return
        t1 = time.time()
        rtt = (t1 - t0) * 1000.0
        mid = (t0 + t1) / 2.0                       # instante médio da viagem
        off = (rep.get("wall", mid) - mid) * 1000.0  # wall remoto − local (ms)
        offs.append(off)
        print(f"  #{i + 1}  RTT={rtt:6.1f} ms   offset(remoto−local)={off:+7.1f} ms", flush=True)
        time.sleep(0.3)
    if offs:
        med = sorted(offs)[len(offs) // 2]
        veredito = "chrony OK (poucos ms)" if abs(med) < 50 else "ATENÇÃO: offset alto — checar chrony!"
        print(f"[ping] offset mediano = {med:+.1f} ms → {veredito}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Pré-teste do trial-sync (ZMQ + chrony)")
    ap.add_argument("--serve", action="store_true", help="sobe o REP mínimo (rodar na bengala)")
    ap.add_argument("--ping", metavar="IP", help="pinga o REP nesse IP (rodar no rpi5-realsense)")
    ap.add_argument("--port", type=int, default=5571, help="porta (default 5571)")
    ap.add_argument("--n", type=int, default=5, help="nº de pings (default 5)")
    a = ap.parse_args()
    if a.serve:
        serve(a.port)
    elif a.ping:
        ping(a.ping, a.port, a.n)
    else:
        ap.error("use --serve OU --ping <ip>")
