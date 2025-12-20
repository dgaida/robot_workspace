"""
Unit tests for cli.py (Command Line Interface)
"""

import pytest
from click.testing import CliRunner
from robot_workspace.cli import cli, info, transform


class TestCLI:
    """Test suite for CLI main group"""

    def test_cli_group_exists(self):
        """Test that CLI group is defined"""
        assert cli is not None
        assert callable(cli)

    def test_cli_help(self):
        """Test CLI help command"""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "Robot Workspace CLI" in result.output

    def test_cli_no_command(self):
        """Test CLI without any command"""
        runner = CliRunner()
        result = runner.invoke(cli, [])

        # Should show help or usage information
        assert result.exit_code == 0

    def test_cli_invalid_command(self):
        """Test CLI with invalid command"""
        runner = CliRunner()
        result = runner.invoke(cli, ["invalid_command"])

        # Should fail with error
        assert result.exit_code != 0
        # Click usually shows "No such command" error
        assert "No such command" in result.output or "Error" in result.output


class TestInfoCommand:
    """Test suite for info command"""

    def test_info_command_exists(self):
        """Test that info command is defined"""
        assert info is not None
        assert callable(info)

    def test_info_help(self):
        """Test info command help"""
        runner = CliRunner()
        result = runner.invoke(cli, ["info", "--help"])

        assert result.exit_code == 0
        assert "info" in result.output.lower()
        assert "workspace" in result.output.lower()

    def test_info_default_workspace(self):
        """Test info command with default workspace"""
        runner = CliRunner()
        result = runner.invoke(cli, ["info"])

        # Command should run (may not do anything without implementation)
        # Exit code should be 0 even if no implementation
        assert result.exit_code == 0

    def test_info_with_workspace_id(self):
        """Test info command with workspace ID option"""
        runner = CliRunner()
        result = runner.invoke(cli, ["info", "--workspace-id", "test_ws"])

        assert result.exit_code == 0

    def test_info_with_custom_workspace(self):
        """Test info command with custom workspace"""
        runner = CliRunner()
        result = runner.invoke(cli, ["info", "--workspace-id", "custom_workspace"])

        assert result.exit_code == 0

    def test_info_workspace_id_option_exists(self):
        """Test that workspace-id option is recognized"""
        runner = CliRunner()
        result = runner.invoke(cli, ["info", "--help"])

        assert "--workspace-id" in result.output

    def test_info_multiple_calls(self):
        """Test calling info command multiple times"""
        runner = CliRunner()

        result1 = runner.invoke(cli, ["info", "--workspace-id", "ws1"])
        result2 = runner.invoke(cli, ["info", "--workspace-id", "ws2"])

        assert result1.exit_code == 0
        assert result2.exit_code == 0


class TestTransformCommand:
    """Test suite for transform command"""

    def test_transform_command_exists(self):
        """Test that transform command is defined"""
        assert transform is not None
        assert callable(transform)

    def test_transform_help(self):
        """Test transform command help"""
        runner = CliRunner()
        result = runner.invoke(cli, ["transform", "--help"])

        assert result.exit_code == 0
        assert "transform" in result.output.lower()
        assert "input" in result.output.lower()
        assert "output" in result.output.lower()

    def test_transform_missing_input(self):
        """Test transform command without input"""
        runner = CliRunner()
        result = runner.invoke(cli, ["transform"])

        # Should fail - input is required
        assert result.exit_code != 0
        assert "input" in result.output.lower() or "required" in result.output.lower()

    def test_transform_missing_output(self):
        """Test transform command without output"""
        runner = CliRunner()
        result = runner.invoke(cli, ["transform", "--input", "input.json"])

        # Should fail - output is required
        assert result.exit_code != 0
        assert "output" in result.output.lower() or "required" in result.output.lower()

    def test_transform_with_both_arguments(self):
        """Test transform command with both input and output"""
        runner = CliRunner()
        result = runner.invoke(cli, ["transform", "--input", "input.json", "--output", "output.json"])

        # Command should execute (may do nothing without implementation)
        assert result.exit_code == 0

    def test_transform_input_option_exists(self):
        """Test that input option is recognized"""
        runner = CliRunner()
        result = runner.invoke(cli, ["transform", "--help"])

        assert "--input" in result.output

    def test_transform_output_option_exists(self):
        """Test that output option is recognized"""
        runner = CliRunner()
        result = runner.invoke(cli, ["transform", "--help"])

        assert "--output" in result.output

    def test_transform_with_file_paths(self):
        """Test transform with various file paths"""
        runner = CliRunner()

        # Test with absolute path style
        result = runner.invoke(cli, ["transform", "--input", "/path/to/input.json", "--output", "/path/to/output.json"])

        assert result.exit_code == 0

    def test_transform_with_relative_paths(self):
        """Test transform with relative paths"""
        runner = CliRunner()

        result = runner.invoke(cli, ["transform", "--input", "input.json", "--output", "output.json"])

        assert result.exit_code == 0

    def test_transform_with_different_extensions(self):
        """Test transform with different file extensions"""
        runner = CliRunner()

        result = runner.invoke(cli, ["transform", "--input", "data.json", "--output", "result.json"])

        assert result.exit_code == 0

    def test_transform_short_options_if_exist(self):
        """Test if short options exist for transform"""
        runner = CliRunner()
        result = runner.invoke(cli, ["transform", "--help"])

        # Check if help shows command structure
        assert "input" in result.output.lower()
        assert "output" in result.output.lower()


class TestCLIIntegration:
    """Integration tests for CLI commands"""

    def test_all_commands_in_help(self):
        """Test that all commands appear in main help"""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "info" in result.output
        assert "transform" in result.output

    def test_command_order_in_group(self):
        """Test that commands are registered in the group"""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])

        # Both commands should be listed
        output_lower = result.output.lower()
        assert "info" in output_lower
        assert "transform" in output_lower

    def test_help_available_for_all_commands(self):
        """Test that help is available for all commands"""
        runner = CliRunner()

        # Test main help
        result_main = runner.invoke(cli, ["--help"])
        assert result_main.exit_code == 0

        # Test info help
        result_info = runner.invoke(cli, ["info", "--help"])
        assert result_info.exit_code == 0

        # Test transform help
        result_transform = runner.invoke(cli, ["transform", "--help"])
        assert result_transform.exit_code == 0


class TestCLIErrorHandling:
    """Test error handling in CLI"""

    def test_invalid_option_info(self):
        """Test info command with invalid option"""
        runner = CliRunner()
        result = runner.invoke(cli, ["info", "--invalid-option"])

        assert result.exit_code != 0
        assert "invalid" in result.output.lower() or "no such option" in result.output.lower()

    def test_invalid_option_transform(self):
        """Test transform command with invalid option"""
        runner = CliRunner()
        result = runner.invoke(cli, ["transform", "--invalid-option"])

        assert result.exit_code != 0

    def test_empty_string_workspace_id(self):
        """Test info command with empty string workspace ID"""
        runner = CliRunner()
        result = runner.invoke(cli, ["info", "--workspace-id", ""])

        # Should accept empty string (validation would be in implementation)
        assert result.exit_code == 0

    def test_empty_string_input(self):
        """Test transform with empty input string"""
        runner = CliRunner()
        result = runner.invoke(cli, ["transform", "--input", "", "--output", "output.json"])

        # Should accept empty string (validation would be in implementation)
        assert result.exit_code == 0

    def test_whitespace_in_paths(self):
        """Test transform with paths containing whitespace"""
        runner = CliRunner()
        result = runner.invoke(cli, ["transform", "--input", "my input.json", "--output", "my output.json"])

        assert result.exit_code == 0


class TestCLIOutputFormatting:
    """Test output formatting of CLI commands"""

    def test_help_output_formatting(self):
        """Test that help output is properly formatted"""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        # Help should contain usage information
        assert "Usage:" in result.output or "usage:" in result.output.lower()

    def test_info_help_describes_workspace_option(self):
        """Test that info help describes workspace-id option"""
        runner = CliRunner()
        result = runner.invoke(cli, ["info", "--help"])

        assert result.exit_code == 0
        output_lower = result.output.lower()
        assert "workspace" in output_lower

    def test_transform_help_describes_io_options(self):
        """Test that transform help describes input/output options"""
        runner = CliRunner()
        result = runner.invoke(cli, ["transform", "--help"])

        assert result.exit_code == 0
        output_lower = result.output.lower()
        assert "input" in output_lower
        assert "output" in output_lower


class TestCLIEdgeCases:
    """Test edge cases in CLI"""

    def test_very_long_workspace_id(self):
        """Test info with very long workspace ID"""
        runner = CliRunner()
        long_id = "a" * 1000
        result = runner.invoke(cli, ["info", "--workspace-id", long_id])

        # Should handle long strings (even if impractical)
        assert result.exit_code == 0

    def test_special_characters_in_workspace_id(self):
        """Test info with special characters in workspace ID"""
        runner = CliRunner()
        result = runner.invoke(cli, ["info", "--workspace-id", "test@workspace#123"])

        assert result.exit_code == 0

    def test_unicode_in_file_paths(self):
        """Test transform with unicode in file paths"""
        runner = CliRunner()
        result = runner.invoke(cli, ["transform", "--input", "входные_данные.json", "--output", "выходные_данные.json"])

        assert result.exit_code == 0

    def test_repeated_options(self):
        """Test command with repeated options (last one should win)"""
        runner = CliRunner()
        result = runner.invoke(cli, ["info", "--workspace-id", "first", "--workspace-id", "second"])

        # Click typically uses the last value
        assert result.exit_code == 0

    def test_mixed_case_options(self):
        """Test that option names are case-sensitive"""
        runner = CliRunner()
        result = runner.invoke(cli, ["info", "--Workspace-Id", "test"])

        # Should fail - options are case-sensitive
        assert result.exit_code != 0


class TestCLIAsModule:
    """Test running CLI as module"""

    def test_cli_callable_directly(self):
        """Test that CLI can be called directly"""
        runner = CliRunner()
        result = runner.invoke(cli)

        assert result.exit_code == 0

    def test_info_callable_directly(self):
        """Test that info can be called directly"""
        runner = CliRunner()
        result = runner.invoke(info)

        assert result.exit_code == 0

    def test_transform_fails_without_required_args(self):
        """Test that transform fails without required arguments"""
        runner = CliRunner()
        result = runner.invoke(transform)

        # Should fail - missing required arguments
        assert result.exit_code != 0


class TestCLIDocumentation:
    """Test CLI documentation strings"""

    def test_cli_has_docstring(self):
        """Test that CLI group has docstring"""
        assert cli.__doc__ is not None or cli.help is not None

    def test_info_has_description(self):
        """Test that info command has description"""
        runner = CliRunner()
        result = runner.invoke(cli, ["info", "--help"])

        # Should have some description
        assert result.exit_code == 0
        assert len(result.output) > 50  # Should have substantial help text

    def test_transform_has_description(self):
        """Test that transform command has description"""
        runner = CliRunner()
        result = runner.invoke(cli, ["transform", "--help"])

        assert result.exit_code == 0
        assert len(result.output) > 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
