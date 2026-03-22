"""
serial_read.py
--------------
Leitura de dados enviados pelo Arduino via porta serial (COM).
Os dados recebidos são salvos em arquivos .txt na pasta data/raw/.

Uso
---
    python src/serial_read.py --port COM3 --baud 9600 --output data/raw/arduino_leitura.txt

    # Listar portas disponíveis
    python src/serial_read.py --list-ports
"""

import os
import time
import argparse

# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------

def list_available_ports():
    """Lista todas as portas seriais disponíveis no sistema."""
    try:
        import serial.tools.list_ports
        ports = list(serial.tools.list_ports.comports())
        if not ports:
            print("Nenhuma porta serial encontrada.")
        for p in ports:
            print(f"  {p.device:10s} — {p.description}")
    except ImportError:
        print("pyserial não instalado. Execute: pip install pyserial")


def combine_txt_files(input_dir: str, output_file: str):
    """
    Combina todos os arquivos .txt de uma pasta em um único arquivo.
    (Equivalente ao auxiliar.py original.)
    """
    try:
        files = sorted(f for f in os.listdir(input_dir) if f.endswith(".txt"))
        if not files:
            print(f"  [serial] Nenhum .txt encontrado em: {input_dir}")
            return

        with open(output_file, "w", encoding="utf-8") as out:
            for fname in files:
                fpath = os.path.join(input_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        content = f.read()
                except UnicodeDecodeError:
                    with open(fpath, "r", encoding="latin1") as f:
                        content = f.read()

                out.write(f"--- Início: {fname} ---\n")
                out.write(content)
                out.write(f"\n--- Fim: {fname} ---\n\n")

        print(f"  [serial] Arquivos combinados em: {output_file}")
    except Exception as e:
        print(f"  [serial] Erro ao combinar arquivos: {e}")


# ---------------------------------------------------------------------------
# Leitura serial
# ---------------------------------------------------------------------------

def read_serial(
    port: str,
    baud: int,
    output_path: str,
    timeout: float = 1.0,
    duration: float = None,
    encoding: str = "utf-8",
):
    """
    Lê dados da porta serial e salva em arquivo.

    Parâmetros
    ----------
    port        : str   — porta serial (ex.: 'COM3' ou '/dev/ttyUSB0')
    baud        : int   — taxa de transmissão (ex.: 9600)
    output_path : str   — arquivo de destino
    timeout     : float — timeout de leitura em segundos
    duration    : float — duração máxima da leitura (None = até Ctrl+C)
    encoding    : str   — codificação para decodificar os bytes recebidos
    """
    try:
        import serial
    except ImportError:
        print("pyserial não instalado. Execute: pip install pyserial")
        return

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    print(f"  [serial] Conectando em {port} @ {baud} baud...")
    try:
        ser = serial.Serial(port, baud, timeout=timeout)
    except serial.SerialException as e:
        print(f"  [serial] Erro ao abrir porta: {e}")
        return

    print(f"  [serial] Lendo dados. Saída: {output_path}")
    print("  [serial] Pressione Ctrl+C para encerrar.\n")

    start = time.time()
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            while True:
                if duration and (time.time() - start) > duration:
                    print("\n  [serial] Duração máxima atingida.")
                    break
                line = ser.readline()
                if line:
                    decoded = line.decode(encoding, errors="replace").strip()
                    print(decoded)
                    f.write(decoded + "\n")
    except KeyboardInterrupt:
        print("\n  [serial] Leitura interrompida pelo usuário.")
    finally:
        ser.close()
        print(f"  [serial] Porta {port} fechada.")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Leitura serial do Arduino")
    parser.add_argument("--list-ports", action="store_true",
                        help="Lista portas seriais disponíveis e sai")
    parser.add_argument("--port",   default="COM3",
                        help="Porta serial (default: COM3)")
    parser.add_argument("--baud",   type=int, default=9600,
                        help="Taxa de transmissão (default: 9600)")
    parser.add_argument("--output", default="data/raw/arduino_leitura.txt",
                        help="Arquivo de saída")
    parser.add_argument("--duration", type=float, default=None,
                        help="Duração máxima da leitura em segundos")
    parser.add_argument("--combine-dir", default=None,
                        help="Combina todos os .txt de um diretório em um único arquivo")
    parser.add_argument("--combine-out", default="data/raw/arduino_combinado.txt",
                        help="Arquivo de saída ao usar --combine-dir")
    args = parser.parse_args()

    if args.list_ports:
        list_available_ports()
    elif args.combine_dir:
        combine_txt_files(args.combine_dir, args.combine_out)
    else:
        read_serial(
            port=args.port,
            baud=args.baud,
            output_path=args.output,
            duration=args.duration,
        )
