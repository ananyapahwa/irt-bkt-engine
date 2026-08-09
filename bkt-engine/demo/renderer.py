try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

def render_table(headers, rows):
    if HAS_TABULATE:
        print(tabulate(rows, headers=headers, tablefmt="fancy_grid"))
    else:
        # Fallback
        header_str = " | ".join(f"{h:<15}" for h in headers)
        print(header_str)
        print("-" * len(header_str))
        for row in rows:
            print(" | ".join(f"{str(x):<15}" for x in row))
