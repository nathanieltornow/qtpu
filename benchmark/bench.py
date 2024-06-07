import argparse





def main():
    parser = argparse.ArgumentParser(description="Benchmarking tool for the project")
    parser.add_argument("--benches", type=str, nargs="+", help="Benchmark names to run")
    parser.add_argument(
        "--qrange", type=int, nargs="+", help="Qubit range for the benchmarks"
    )
    
    pass


if __name__ == "__main__":
    main()
