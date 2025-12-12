# robot_workspace/cli.py
import click

# from robot_workspace import NiryoWorkspaces, Objects


@click.group()
def cli():
    """Robot Workspace CLI"""
    pass


@cli.command()
@click.option("--workspace-id", default="niryo_ws", help="Workspace ID")
def info(workspace_id):
    """Display workspace information"""
    # Implementation
    pass


@cli.command()
@click.option("--input", required=True, help="Input JSON file")
@click.option("--output", required=True, help="Output JSON file")
def transform(input, output):
    """Transform objects between formats"""
    # Implementation
    pass


if __name__ == "__main__":
    cli()
