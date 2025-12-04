"""
Integration testing with the CLI

"""

from cli.cli import cli
from click.testing import CliRunner
from PIL import Image


def test_help():
    """Tests the command-line interface help message."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Show this message and exit." in result.output


def test_predict_cli():
    """Probar que el comando predict funciona con un archivo real."""
    runner = CliRunner()
    # Usamos un sistema de archivos aislado para no ensuciar tu carpeta con imágenes de prueba
    with runner.isolated_filesystem():
        # 1. Crear una imagen dummy temporal
        img = Image.new("RGB", (60, 30), color="red")
        img.save("test_image.jpg")

        # 2. Invocar el comando predict pasándole la imagen creada
        result = runner.invoke(cli, ["predict", "test_image.jpg"])

        # 3. Verificar resultado
        assert result.exit_code == 0
        assert "La imagen test_image.jpg es un:" in result.output


def test_resize_cli():
    """Probar que el comando resize funciona."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        # 1. Crear imagen dummy
        img = Image.new("RGB", (60, 30), color="red")
        img.save("test_image.jpg")

        # 2. Invocar resize (width=50, height=50)
        result = runner.invoke(cli, ["resize", "test_image.jpg", "50", "50"])

        assert result.exit_code == 0
        # El mensaje de éxito debe contener las nuevas dimensiones
        assert "Imagen redimensionada a: (50, 50)" in result.output


def test_grayscale_cli():
    """Probar comando grayscale."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        # 1. Crear imagen color
        img = Image.new("RGB", (60, 30), color="blue")
        img.save("test_image.jpg")

        # 2. Invocar grayscale
        result = runner.invoke(cli, ["grayscale", "test_image.jpg"])

        assert result.exit_code == 0
        # Verificamos que diga "Modo: L" (L es escala de grises en PIL)
        assert "Modo: L" in result.output


def test_flatten_cli():
    """Probar comando flatten."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        # 1. Crear imagen de 10x10 píxeles (total 100)
        img = Image.new("RGB", (10, 10), color="green")
        img.save("test_image.jpg")

        # 2. Invocar flatten
        result = runner.invoke(cli, ["flatten", "test_image.jpg"])

        assert result.exit_code == 0
        # 10 * 10 = 100 píxeles
        assert "Total píxeles: 100" in result.output
