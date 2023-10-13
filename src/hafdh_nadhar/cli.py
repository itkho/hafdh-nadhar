from typing import Annotated, Optional

import typer

from hafdh_nadhar.hafdh import hafdh_img

app = typer.Typer(add_completion=False)


@app.command()
def hafdh(
    img_path: str,
    open_in_window: Annotated[bool, typer.Option(help="Open the result image")] = True,
    save_to_path: Annotated[
        Optional[str], typer.Option(help="Path to save the result image")
    ] = None,
):
    """
    Remove human representations from an image by blurring them.
    """
    hafdh_img(
        img_path=img_path, open_in_window=open_in_window, save_to_path=save_to_path
    )


if __name__ == "__main__":
    app()
