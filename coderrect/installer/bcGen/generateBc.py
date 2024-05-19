import argparse
import yaml
import os
import subprocess
import pathlib
import tempfile
import os
from typing import List
import shutil
def install_coderrect(version, install_dir = './coderrect'):
    install_path = pathlib.Path(install_dir)
    install_path.mkdir(exist_ok=True)
    tarfile = 'coderrect-linux-{}.tar.gz'.format(version)
    if not install_path.joinpath(tarfile).exists():
        subprocess.run(['wget', 'https://public-installer-pkg.s3.us-east-2.amazonaws.com/{}'.format(tarfile)], cwd=install_path)
        subprocess.run(['tar', '-xf', tarfile], cwd=install_path)
    return install_path.joinpath('coderrect-linux-{}/bin'.format(version)).resolve()
class Project:
    def __init__(self, name, fetch, pre_build, pre_build_dir, build_dir, versions, build, clean):
        self.name = name
        self.fetch = fetch
        self.pre_build = pre_build
        self.pre_build_dir = pre_build_dir
        self.build_dir = build_dir
        self.versions = versions
        self.build = build
        self.clean = clean
def load_project(project, yaml_dict) -> Project:
    data = yaml_dict[project]
    name = project
    fetch = data['fetch']
    pre_build = None if 'pre_build' not in data else data['pre_build']
    pre_build_dir = None if 'pre_build_dir' not in data else data['pre_build_dir']
    build_dir = data['build_dir']
    versions = data['versions']
    build = data['build']
    clean = data['clean']
    return Project(name, fetch, pre_build, pre_build_dir, build_dir, versions, build, clean)
class Tool:
    def __init__(self, version, path):
        self.version = version
        self.path = path
def build_it(project: Project, tools: List[Tool], workdir='./sources', logdir = './logs', outdir = './benchmarks'):
    workdir = pathlib.Path(workdir).joinpath(project.name)
    workdir.mkdir(exist_ok=True, parents=True)
    logdir = pathlib.Path(logdir)
    logdir.mkdir(exist_ok=True)
    outdir = pathlib.Path(outdir)
    outdir.mkdir(exist_ok=True)
    for version in project.versions:
        vdir = workdir.joinpath(version)
        vdir.mkdir(exist_ok=True)
        # Fetch
        subprocess.run(project.fetch, cwd=vdir, shell=True)
        # Build Setup
        if project.pre_build:
            logstdout = logdir.joinpath(project.name, 'prebuildout.txt')
            logstderr = logdir.joinpath(project.name, 'prebuilderr.txt')
            logstdout.parent.mkdir(exist_ok=True, parents=True)
            logstderr.parent.mkdir(exist_ok=True, parents=True)
            subprocess.run([project.pre_build], 
                            shell=True, 
                            cwd=vdir.joinpath(project.pre_build_dir), 
                            stdout=open(logstdout, 'w'), 
                            stderr=open(logstderr, 'w')
            )
        # Build
        builddir = vdir.joinpath(project.build_dir)
        subprocess.run(['git', 'checkout', version], cwd=builddir)
        for tool in tools:
            subprocess.run(project.clean, cwd=builddir, shell=True)
            if builddir.joinpath('.coderrect').exists():
                shutil.rmtree(builddir.joinpath('.coderrect'))
            subenv = os.environ.copy()
            subenv['PATH'] = str(tool.path)+':'+subenv['PATH']
            logstdout = logdir.joinpath(project.name, version, tool.version, 'out.txt')
            logstderr = logdir.joinpath(project.name, version, tool.version, 'err.txt')
            logstdout.parent.mkdir(exist_ok=True, parents=True)
            logstderr.parent.mkdir(exist_ok=True, parents=True)
            subprocess.run(['coderrect -XbcOnly ' + project.build], cwd=builddir, shell=True, env=subenv, stdout=open(logstdout, 'w'), stderr=open(logstderr, 'w'))
            # Copy bc files
            out = outdir.joinpath(project.name, version, tool.version)
            out.mkdir(parents=True, exist_ok=True)
            for bc in pathlib.Path(builddir).rglob('*.bc'):
                if bc.name.startswith('.') or bc.name.endswith('.o.bc'):
                    continue
                bc.rename(out.joinpath(bc.name))
    # # Fetch
    # subprocess.run(project.fetch, cwd=workdir, shell=True)
    # # Build Setup
    # if project.pre_build:
    #     logstdout = logdir.joinpath(project.name, 'prebuildout.txt')
    #     logstderr = logdir.joinpath(project.name, 'prebuilderr.txt')
    #     logstdout.parent.mkdir(exist_ok=True, parents=True)
    #     logstderr.parent.mkdir(exist_ok=True, parents=True)
    #     subprocess.run([project.pre_build], 
    #                     shell=True, 
    #                     cwd=workdir.joinpath(project.pre_build_dir), 
    #                     stdout=open(logstdout, 'w'), 
    #                     stderr=open(logstderr, 'w')
    #     )
    # # Build
    # builddir = workdir.joinpath(project.build_dir)
    # for version in project.versions:
    #     subprocess.run(['git', 'checkout', version], cwd=builddir)
    #     for tool in tools:
    #         subprocess.run(project.clean, cwd=builddir, shell=True)
    #         if builddir.joinpath('.coderrect').exists():
    #             shutil.rmtree(builddir.joinpath('.coderrect'))
    #         subenv = os.environ.copy()
    #         subenv['PATH'] = str(tool.path)+':'+subenv['PATH']
    #         logstdout = logdir.joinpath(project.name, version, tool.version, 'out.txt')
    #         logstderr = logdir.joinpath(project.name, version, tool.version, 'err.txt')
    #         logstdout.parent.mkdir(exist_ok=True, parents=True)
    #         logstderr.parent.mkdir(exist_ok=True, parents=True)
    #         subprocess.run(['coderrect -XbcOnly ' + project.build], cwd=builddir, shell=True, env=subenv, stdout=open(logstdout, 'w'), stderr=open(logstderr, 'w'))
    #         # Copy bc files
    #         out = outdir.joinpath(project.name, version, tool.version)
    #         out.mkdir(parents=True, exist_ok=True)
    #         for bc in pathlib.Path(builddir).rglob('*.bc'):
    #             if bc.name.startswith('.') or bc.name.endswith('.o.bc'):
    #                 continue
    #             bc.rename(out.joinpath(bc.name))
def main():
    parser = argparse.ArgumentParser(description="Build the bc for all benchmarks")
    parser.add_argument('-o', '--out', dest='out', type=str, help='output directory to store generated bcs')
    parser.add_argument('-p', '--project', dest='project', type=str, help='specify single project to build')
    args = parser.parse_args()
    tools = []
    tools.append(Tool('9', install_coderrect('hpc-0.8.5')))
    tools.append(Tool('10', install_coderrect('develop')))
    try:
        with open('benchmarks.yaml', 'r') as f:
            benchmarks = yaml.load(f, Loader=yaml.FullLoader)
    except yaml.parser.ParserError as e:
        print("Error parsing benchmarks.yaml:", e)
        return 1
    project_names = []
    if args.project:
        if args.project not in benchmarks.keys():
            print('no entry for', args.project)
            return 1
        project_names = [args.project]
    else:
        project_names  = benchmarks.keys()
    projects = []
    for name in project_names:
        proj = load_project(name, benchmarks)
        projects.append(proj)
    for project in projects:
        print(project.name)
        print(project.versions)
        build_it(project, tools)
if __name__ == "__main__":
    main()
