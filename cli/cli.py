"""
Main CLI or app entry point
"""

import click
from PIL import Image
from mylib.logic import predict, resize, convert_to_grayscale, flatten_image


# We create a group of commands
@click.group()
def cli():
    """Main CLI to perform arithmetical operations."""


@cli.command("predict")
@click.argument("filepath", type=click.Path(exists=True))
def predict_cli(filepath):
    """Predecir clase de un archivo de imagen."""
    image = Image.open(filepath)
    result = predict(image)
    click.echo(click.style(f"La imagen {filepath} es un: {result}", fg="green"))


@cli.command("resize")
@click.argument("filepath", type=click.Path(exists=True))
@click.argument("width", type=int)
@click.argument("height", type=int)
def resize_cli(filepath, width, height):
    """Redimensionar imagen."""
    image = Image.open(filepath)
    new_img = resize(image, width, height)
    click.echo(click.style(f"Imagen redimensionada a: {new_img.size}", fg="green"))


@cli.command("grayscale")
@click.argument("filepath", type=click.Path(exists=True))
def grayscale_cli(filepath):
    """Convertir imagen a escala de grises."""
    image = Image.open(filepath)
    new_img = convert_to_grayscale(image)
    click.echo(
        click.style(
            f"Imagen convertida a escala de grises. Modo: {new_img.mode}", fg="green"
        )
    )


@cli.command("flatten")
@click.argument("filepath", type=click.Path(exists=True))
def flatten_cli(filepath):
    """Aplanar imagen a una lista de píxeles."""
    image = Image.open(filepath)
    data = flatten_image(image)
    click.echo(click.style(f"Imagen aplanada. Total píxeles: {len(data)}", fg="green"))


if __name__ == "__main__":
    cli()
