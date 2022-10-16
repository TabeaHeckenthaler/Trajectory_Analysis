from os import path, mkdir
import datetime
from Directories import project_home


def graph_dir():
    direct = path.abspath(path.join(project_home, 'Graphs',
                                    datetime.datetime.now().strftime("%Y") + '_' +
                                    datetime.datetime.now().strftime("%m") + '_' +
                                    datetime.datetime.now().strftime("%d")))
    if not (path.isdir(direct)):
        mkdir(direct)
    return direct


def save_fig(fig, name):
    name = "".join(x for x in name if x.isalnum())
    if fig.__module__ == 'plotly.graph_objs._figure':
        fig.write_image(graph_dir() + path.sep + name + '.pdf')
        fig.write_image(graph_dir() + path.sep + name + '.svg')
    else:
        fig.savefig(graph_dir() + path.sep + name + '.pdf', format='pdf', pad_inches=0.5, bbox_inches='tight')
        fig.savefig(graph_dir() + path.sep + name + '.svg', format='svg', pad_inches=0.5, bbox_inches='tight')