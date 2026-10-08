import click
import pandas as pd
from pathlib import Path


@click.command()
@click.option("--input", "input_path", type=click.Path(exists=True, path_type=Path), required=True)
def main(input_path: Path):
    df = pd.read_csv(input_path)
    print(f"Shape: {df.shape}")
    print(f"\nColumns:\n{list(df.columns)}")
    print(f"\nDtypes:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nDescriptive stats:\n{df.describe()}")
    if "label" in df.columns:
        print(f"\nClass distribution:\n{df['label'].value_counts()}")


if __name__ == "__main__":
    main()
