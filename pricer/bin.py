import click

from pricer.ui import index

@click.group()
def cli():
    pass

@click.command()
@click.option('--port', default=8887, help='port')
def server(port):
    index.start(port)

cli.add_command(server)

if __name__ == '__main__':
    cli()
