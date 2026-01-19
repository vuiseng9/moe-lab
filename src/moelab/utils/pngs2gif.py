from PIL import Image
import glob
from pathlib import Path
import typer
import math

app = typer.Typer()


@app.command()
def pngs2gif(
    folder: str = typer.Argument(..., help="Folder containing PNG files"),
    output: str = typer.Option("output.gif", "--output", "-o", help="Output GIF filename"),
    downsample: int = typer.Option(None, "--downsample", "-d", help="Keep every Nth frame"),
    max_len: float = typer.Option(None, "--max-len", "-m", help="Maximum duration in seconds"),
):
    """Convert all PNG files in a folder to an animated GIF."""
    if downsample is not None and max_len is not None:
        typer.echo("Error: Cannot use both --downsample and --max-len", err=True)
        raise typer.Exit(1)
    
    folder_path = Path(folder)
    
    if not folder_path.exists():
        typer.echo(f"Error: Folder '{folder}' does not exist", err=True)
        raise typer.Exit(1)
    
    png_pattern = str(folder_path / "*.png")
    png_files = sorted(glob.glob(png_pattern))
    
    if not png_files:
        typer.echo(f"Error: No PNG files found in '{folder}'", err=True)
        raise typer.Exit(1)
    
    total_files = len(png_files)
    frame_duration = 0.1  # 100ms = 0.1 seconds
    
    # Calculate downsampling factor
    if max_len is not None:
        desired_frames = int(max_len / frame_duration)
        downsample = max(1, math.ceil(total_files / desired_frames))
        typer.echo(f"Target duration: {max_len}s → calculated downsample factor: {downsample}")
    elif downsample is None:
        downsample = 1
    
    # Downsample: keep every Nth file
    png_files = [f for i, f in enumerate(png_files) if i % downsample == 0]
    
    typer.echo(f"Found {len(png_files)} PNG files {f"(downsampled, keep every {downsample}th)." if downsample > 1 else ""}")
    frames = [Image.open(f) for f in png_files]
    
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=100,  # ms per frame
        loop=0
    )
    
    typer.echo(f"Created {output}")


if __name__ == "__main__":
    app()
