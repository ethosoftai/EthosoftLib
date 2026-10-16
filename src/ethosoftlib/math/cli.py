import argparse
import json
import sys

from . import (__version__, calculate, convert_text, execute, fill_results, render_protocol_md,
               tool_use, verify_converted)


def _convert_file(args):
    n = n_conv = n_calls = n_bad = 0
    with open(args.input, encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = row.get(args.field)
            if isinstance(text, str):
                out, _ = convert_text(text, aggressive=args.aggressive, lang=args.lang)
                calls, bad = verify_converted(out)
                row[args.out_field or args.field] = out
                row["tool_calls"] = calls
                n_calls += calls
                n_bad += bad
                n_conv += calls > 0
            n += 1
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"satir={n} aracli={n_conv} ({100 * n_conv / max(1, n):.1f}%) cagri={n_calls} hatali={n_bad}")


def _translate(args):
    from .translate import MODEL_ID, MathTranslator
    t = MathTranslator(args.model or MODEL_ID)
    if args.input:
        if not args.output:
            raise SystemExit("--input ile --output de gerekli")
        fields = tuple(args.fields.split(","))
        n = ok = 0
        with open(args.input, encoding="utf-8") as fin, open(args.output, "w", encoding="utf-8") as fout:
            for line in fin:
                if not line.strip():
                    continue
                n += 1
                new = t.translate_record(json.loads(line), fields=fields)
                if new is not None:
                    ok += 1
                    fout.write(json.dumps(new, ensure_ascii=False) + "
")
        print(f"satir={n} cevrilen={ok} atilan={n - ok}")
    else:
        print(t.translate(args.text if args.text is not None else sys.stdin.read()))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ethosoftlib-math", description="Matematik tool-use protokolu")
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("convert", help="jsonl dosyasindaki bir alani tool-use bicimine cevir")
    c.add_argument("input")
    c.add_argument("output")
    c.add_argument("--field", default="solution")
    c.add_argument("--out-field", default=None)
    c.add_argument("--aggressive", action="store_true", help="'=' olmayan sayisal islemleri de cevir")
    c.add_argument("--lang", choices=["tr", "en"], default="tr", help="etiket/fonksiyon adi dili")
    c.set_defaults(func=_convert_file)

    p = sub.add_parser("protocol", help="protokol dokumanini yazdir (Markdown)")
    p.set_defaults(func=lambda a: print(render_protocol_md()))

    k = sub.add_parser("calc", help="serbest ifadeyi hesapla, orn: ethosoftlib-math calc '4 + 5'")
    k.add_argument("expr")
    k.add_argument("--trace", action="store_true", help="arac cagrisi izini de goster")
    k.set_defaults(func=lambda a: print(calculate(a.expr, trace=a.trace)))

    f = sub.add_parser("fill", help="metindeki <arac_cagrisi> bloklarini calistirip <arac_sonucu> yaz")
    f.add_argument("text", nargs="?", help="verilmezse stdin okunur")
    f.set_defaults(func=lambda a: print(fill_results(a.text if a.text is not None else sys.stdin.read())))

    t = sub.add_parser("tool-use", help="duz metni tool-use bicimine cevir")
    t.add_argument("text")
    t.set_defaults(func=lambda a: print(tool_use(a.text)))

    e = sub.add_parser("exec", help="tek bir arac cagrisini calistir, orn: topla(3, 4)")
    e.add_argument("call")
    e.set_defaults(func=lambda a: print(execute(a.call)))

    tr = sub.add_parser("translate", help="Ingilizce matematik metnini Turkceye cevir (Ethosoft-Math-Translator)")
    tr.add_argument("text", nargs="?", help="verilmezse stdin okunur")
    tr.add_argument("--input", help="jsonl (problem/solution) -> --output'a problem_tr/solution_tr yazar")
    tr.add_argument("--output")
    tr.add_argument("--model", default=None)
    tr.add_argument("--fields", default="problem,solution")
    tr.set_defaults(func=_translate)

    args = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    args.func(args)


if __name__ == "__main__":
    main()
