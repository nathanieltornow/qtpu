import nox
import argparse

BUILD_REQUIREMENTS = [
    "setuptools>=66.1",
    "setuptools_scm>=8.1",
    "wheel>=0.40",
]


@nox.session(reuse_venv=True)
def docs(session: nox.Session) -> None:
    """Build the docs. Use "--non-interactive" to avoid serving. Pass "-b linkcheck" to check links."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-b", dest="builder", default="html", help="Build target (default: html)"
    )
    args, posargs = parser.parse_known_args(session.posargs)

    serve = args.builder == "html" and session.interactive
    extra_installs = ["sphinx-autobuild"] if serve else []
    session.install(*BUILD_REQUIREMENTS, *extra_installs)
    session.install("--no-build-isolation", "-ve.[docs]")
    session.chdir("docs")

    if args.builder == "linkcheck":
        session.run(
            "sphinx-build", "-b", "linkcheck", ".", "_build/linkcheck", *posargs
        )
        return

    shared_args = (
        "-n",  # nitpicky mode
        "-T",  # full tracebacks
        f"-b={args.builder}",
        ".",
        f"_build/{args.builder}",
        *posargs,
    )

    if serve:
        session.run("sphinx-autobuild", "--ignore", "**jupyter**", *shared_args)
    else:
        session.run("sphinx-build", "--keep-going", *shared_args)
