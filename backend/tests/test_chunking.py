from openmfp_copilot.documents.chunking import Section, split_markdown


def test_splits_by_headings_and_keeps_the_heading_path() -> None:
    markdown = (
        "# Portal UI Library\n"
        "Angular library for the portal shell.\n"
        "## Getting started\n"
        "Read this first.\n"
        "### Install\n"
        "npm install @openmfp/portal-ui-lib\n"
        "## License\n"
        "Apache-2.0\n"
    )

    assert split_markdown(markdown, "openmfp/portal-ui-lib") == [
        Section(
            ("openmfp/portal-ui-lib", "Portal UI Library"),
            "Angular library for the portal shell.",
        ),
        Section(
            ("openmfp/portal-ui-lib", "Portal UI Library", "Getting started"),
            "Read this first.",
        ),
        Section(
            ("openmfp/portal-ui-lib", "Portal UI Library", "Getting started", "Install"),
            "npm install @openmfp/portal-ui-lib",
        ),
        Section(("openmfp/portal-ui-lib", "Portal UI Library", "License"), "Apache-2.0"),
    ]


def test_keeps_text_before_the_first_heading_under_the_title() -> None:
    sections = split_markdown("[![build](badge.svg)](ci)\n\n# Portal\nText\n", "openmfp/portal")

    assert sections[0] == Section(("openmfp/portal",), "[![build](badge.svg)](ci)")


def test_skips_headings_without_body() -> None:
    sections = split_markdown("# Title\n## Setup\n### Steps\nRun it.\n", "repo")

    assert sections == [Section(("repo", "Title", "Setup", "Steps"), "Run it.")]


def test_ignores_hash_lines_inside_code_fences() -> None:
    markdown = (
        "## Configure\n"
        "```bash\n"
        "# set the port\n"
        "export FRONTEND_PORT=4300\n"
        "```\n"
        "~~~~\n"
        "## not a heading\n"
        "```\n"
        "~~~~\n"
        "## Next\n"
        "Done.\n"
    )

    [configure, following] = split_markdown(markdown, "repo")

    assert configure.headings == ("repo", "Configure")
    assert "# set the port" in configure.body
    assert "## not a heading" in configure.body
    assert following == Section(("repo", "Next"), "Done.")


def test_strips_closing_hashes_and_indentation_from_headings() -> None:
    [section] = split_markdown("   ## Install ##\nnpm i\n", "repo")

    assert section.headings == ("repo", "Install")


def test_section_text_starts_with_the_heading_path() -> None:
    section = Section(("openmfp/portal-ui-lib", "Getting started", "Install"), "npm i")

    assert section.text == "openmfp/portal-ui-lib > Getting started > Install\n\nnpm i"
