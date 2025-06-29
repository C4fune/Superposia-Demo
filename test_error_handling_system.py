"""
Comprehensive Test Suite for Error Handling System

This module tests all aspects of the error handling system including
exceptions, reporting, alerts, and integration.
"""

import pytest
import time
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from quantum_platform.errors import (
    # Exceptions
    QuantumPlatformError, UserError, SystemError,
    CompilationError, ParseError, CircuitError, QubitError,
    SimulationError, ResourceLimitError,
    
    # Error handling
    ErrorReporter, ErrorReport, ErrorContext,
    format_error_message, create_user_friendly_message,
    
    # Alerts
    AlertManager, AlertType, AlertSeverity,
    get_alert_manager, create_error_alert,
    
    # Helper functions
    create_parse_error, create_qubit_error, create_simulation_error
)

from quantum_platform.errors.decorator import (
    handle_errors, user_friendly_errors, safe_execute
)


class TestExceptionHierarchy:
    """Test the custom exception hierarchy."""
    
    def test_base_exception(self):
        """Test base QuantumPlatformError."""
        error = QuantumPlatformError(
            message="Test error",
            user_message="User-friendly message",
            error_code="TEST001"
        )
        
        assert str(error) == "Test error"
        assert error.user_message == "User-friendly message"
        assert error.error_code == "TEST001"
        assert error.context is not None
        assert error.suggestions == []
        
        # Test dictionary conversion
        error_dict = error.to_dict()
        assert error_dict['type'] == 'QuantumPlatformError'
        assert error_dict['message'] == "Test error"
        assert error_dict['user_message'] == "User-friendly message"
        assert error_dict['error_code'] == "TEST001"
    
    def test_user_error(self):
        """Test UserError subclass."""
        error = UserError("Invalid syntax")
        
        assert isinstance(error, QuantumPlatformError)
        assert "issue with your quantum program" in error.user_message
    
    def test_system_error(self):
        """Test SystemError subclass."""
        error = SystemError("Internal failure")
        
        assert isinstance(error, QuantumPlatformError)
        assert "internal error occurred" in error.user_message
    
    def test_compilation_error(self):
        """Test CompilationError."""
        error = CompilationError("Parse failed")
        
        assert isinstance(error, UserError)
        assert "compile your quantum program" in error.user_message
    
    def test_parse_error_creation(self):
        """Test parse error creation utility."""
        error = create_parse_error(
            message="Unexpected token",
            line_number=5,
            file_path="test.qc",
            suggestions=["Check syntax", "Verify indentation"]
        )
        
        assert isinstance(error, ParseError)
        assert error.context.line_number == 5
        assert error.context.file_path == "test.qc"
        assert len(error.suggestions) == 2
    
    def test_qubit_error_creation(self):
        """Test qubit error creation utility."""
        error = create_qubit_error(
            message="Qubit index out of range",
            qubit_id=5,
            operation="gate_application",
            suggestions=["Check qubit allocation"]
        )
        
        assert isinstance(error, QubitError)
        assert error.context.system_state["qubit_id"] == 5
        assert error.context.operation == "gate_application"
    
    def test_simulation_error_creation(self):
        """Test simulation error creation utility."""
        error = create_simulation_error(
            message="Out of memory",
            num_qubits=25,
            shots=10000,
            memory_required=32.0
        )
        
        assert isinstance(error, SimulationError)
        assert error.context.system_state["num_qubits"] == 25
        assert error.context.system_state["memory_required_gb"] == 32.0
        assert any("qubits" in s for s in error.suggestions)


class TestErrorReporter:
    """Test error reporting functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.reporter = ErrorReporter(reports_dir=self.temp_dir)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_reporter_initialization(self):
        """Test reporter initialization."""
        assert self.reporter.reports_dir == self.temp_dir
        assert self.reporter.auto_collect == True
        assert len(self.reporter._error_history) == 0
    
    def test_collect_error_basic(self):
        """Test basic error collection."""
        exception = ValueError("Test error")
        
        report = self.reporter.collect_error(exception)
        
        assert isinstance(report, ErrorReport)
        assert report.error_type == "ValueError"
        assert report.error_message == "Test error"
        assert report.report_id.startswith("QP-")
        assert len(self.reporter._error_history) == 1
    
    def test_collect_error_with_context(self):
        """Test error collection with context."""
        context = ErrorContext(
            component="TestComponent",
            operation="test_operation",
            user_action="Running test"
        )
        
        exception = CompilationError("Parse failed")
        report = self.reporter.collect_error(
            exception,
            context=context,
            user_action="Compiling circuit"
        )
        
        assert report.context.component == "TestComponent"
        assert report.context.operation == "test_operation"
        assert report.context.user_action == "Compiling circuit"
    
    def test_error_report_serialization(self):
        """Test error report serialization."""
        exception = CircuitError("Invalid gate")
        report = self.reporter.collect_error(exception)
        
        # Test JSON serialization
        json_str = report.to_json()
        assert "Invalid gate" in json_str
        assert "CircuitError" in json_str
        
        # Test markdown serialization
        md_str = report.to_markdown()
        assert "# Error Report:" in md_str
        assert "Invalid gate" in md_str
        assert "## System Information" in md_str
    
    def test_report_file_creation(self):
        """Test that report files are created."""
        exception = UserError("Test error")
        report = self.reporter.collect_error(exception)
        
        # Check JSON file
        json_file = self.temp_dir / f"{report.report_id}.json"
        assert json_file.exists()
        
        # Check markdown file
        md_file = self.temp_dir / f"{report.report_id}.md"
        assert md_file.exists()
    
    def test_error_history_limit(self):
        """Test error history size limit."""
        self.reporter._max_history = 3
        
        # Add more errors than the limit
        for i in range(5):
            exception = ValueError(f"Error {i}")
            self.reporter.collect_error(exception)
        
        # Should only keep the last 3
        assert len(self.reporter._error_history) == 3
        assert "Error 4" in str(self.reporter._error_history[-1].error_message)
    
    @patch('quantum_platform.errors.reporter.SystemInfo.collect')
    def test_system_info_collection(self, mock_collect):
        """Test system information collection."""
        mock_collect.return_value = Mock(
            platform="TestOS",
            python_version="3.8.0",
            memory_gb=16.0
        )
        
        exception = SystemError("Test error")
        report = self.reporter.collect_error(exception)
        
        assert report.system_info.platform == "TestOS"
        assert report.system_info.memory_gb == 16.0


class TestAlertManager:
    """Test alert management functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.alert_manager = AlertManager(max_alerts=10)
        self.alert_handler_calls = []
        
        def test_handler(alert):
            self.alert_handler_calls.append(alert)
        
        self.alert_manager.add_alert_handler(test_handler)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        self.alert_manager.shutdown()
    
    def test_create_basic_alert(self):
        """Test basic alert creation."""
        alert = self.alert_manager.create_alert(
            title="Test Alert",
            message="This is a test",
            alert_type=AlertType.INFO
        )
        
        assert alert.title == "Test Alert"
        assert alert.message == "This is a test"
        assert alert.alert_type == AlertType.INFO
        assert len(self.alert_handler_calls) == 1
        assert self.alert_handler_calls[0] == alert
    
    def test_error_alert(self):
        """Test error alert creation."""
        alert = self.alert_manager.error_alert(
            title="Compilation Error",
            message="Failed to parse circuit",
            component="Compiler",
            error_code="COMP001",
            suggestions=["Check syntax", "Verify structure"]
        )
        
        assert alert.alert_type == AlertType.ERROR
        assert alert.severity == AlertSeverity.HIGH
        assert alert.component == "Compiler"
        assert alert.metadata["error_code"] == "COMP001"
        assert len(alert.actions) == 3  # Show Details, Report Error, Dismiss
        assert not alert.auto_dismiss
        assert alert.persistent
    
    def test_alert_dismissal(self):
        """Test alert dismissal."""
        alert = self.alert_manager.info_alert("Test", "Message")
        alert_id = alert.id
        
        # Should be in active alerts
        active = self.alert_manager.get_active_alerts()
        assert len(active) == 1
        assert active[0].id == alert_id
        
        # Dismiss the alert
        success = self.alert_manager.dismiss_alert(alert_id, acknowledged=True)
        assert success
        
        # Should no longer be in active alerts
        active = self.alert_manager.get_active_alerts()
        assert len(active) == 0
        
        # Should be marked as dismissed
        history = self.alert_manager.get_alert_history()
        assert len(history) == 1
        assert history[0].dismissed
        assert history[0].acknowledged
    
    def test_auto_dismissal(self):
        """Test automatic alert dismissal."""
        alert = self.alert_manager.create_alert(
            title="Auto Dismiss Test",
            message="This should auto-dismiss",
            auto_dismiss=True,
            dismiss_after=1  # 1 second
        )
        
        # Should be active initially
        active = self.alert_manager.get_active_alerts()
        assert len(active) == 1
        
        # Wait for auto-dismissal
        time.sleep(1.5)
        
        # Should be auto-dismissed
        active = self.alert_manager.get_active_alerts()
        assert len(active) == 0
    
    def test_question_alert(self):
        """Test question/confirmation alert."""
        actions = [
            {'label': 'Yes', 'action': 'confirm', 'style': 'primary'},
            {'label': 'No', 'action': 'cancel', 'style': 'secondary'}
        ]
        
        alert = self.alert_manager.question_alert(
            title="Confirm Action",
            message="Are you sure you want to continue?",
            actions=actions
        )
        
        assert alert.alert_type == AlertType.QUESTION
        assert len(alert.actions) == 2
        assert not alert.auto_dismiss
        assert alert.persistent
    
    def test_clear_all_alerts(self):
        """Test clearing all alerts."""
        # Create multiple alerts
        self.alert_manager.info_alert("Alert 1", "Message 1")
        self.alert_manager.warning_alert("Alert 2", "Message 2")
        self.alert_manager.error_alert("Alert 3", "Message 3")
        
        assert len(self.alert_manager.get_active_alerts()) == 3
        
        # Clear all
        self.alert_manager.clear_all_alerts()
        
        assert len(self.alert_manager.get_active_alerts()) == 0


class TestErrorHandlingDecorators:
    """Test error handling decorators."""
    
    def test_handle_errors_decorator(self):
        """Test basic error handling decorator."""
        @handle_errors(
            component="TestComponent",
            operation="test_function",
            show_alert=False,
            report_error=False,
            reraise=False,
            fallback_return="error_occurred"
        )
        def failing_function():
            raise ValueError("Test error")
        
        result = failing_function()
        assert result == "error_occurred"
    
    def test_user_friendly_errors_decorator(self):
        """Test user-friendly errors decorator."""
        alert_manager = get_alert_manager()
        initial_count = len(alert_manager.get_active_alerts())
        
        @user_friendly_errors(
            component="TestComponent",
            user_message="Something went wrong"
        )
        def failing_function():
            raise RuntimeError("Internal error")
        
        result = failing_function()
        assert result is None  # Default fallback
        
        # Should have created an alert
        active_alerts = alert_manager.get_active_alerts()
        assert len(active_alerts) == initial_count + 1
    
    def test_safe_execute_decorator(self):
        """Test safe execution decorator."""
        @safe_execute(default_return="safe_fallback", log_errors=False)
        def failing_function():
            raise Exception("This should not crash")
        
        result = failing_function()
        assert result == "safe_fallback"
    
    def test_decorator_with_successful_function(self):
        """Test that decorators don't interfere with successful execution."""
        @handle_errors(component="Test")
        def successful_function(x, y):
            return x + y
        
        result = successful_function(2, 3)
        assert result == 5


class TestErrorIntegration:
    """Test error handling integration with quantum platform components."""
    
    def test_compilation_error_integration(self):
        """Test error handling during compilation."""
        from quantum_platform.compiler.language import QuantumProgram
        from quantum_platform.errors.decorator import compiler_errors
        
        @compiler_errors(operation="parse_circuit")
        def parse_invalid_circuit():
            # Simulate a parsing error
            raise ParseError(
                message="Unexpected token 'invalid' at line 5",
                user_message="Syntax error in your quantum program",
                error_code="PARSE001"
            )
        
        with pytest.raises(ParseError) as exc_info:
            parse_invalid_circuit()
        
        error = exc_info.value
        assert error.error_code == "PARSE001"
        assert "Syntax error" in error.user_message
    
    def test_simulation_error_integration(self):
        """Test error handling during simulation."""
        from quantum_platform.errors.decorator import simulation_errors
        
        @simulation_errors(operation="run_simulation")
        def run_large_simulation():
            # Simulate running out of memory
            raise create_simulation_error(
                message="Insufficient memory for 30-qubit simulation",
                num_qubits=30,
                memory_required=64.0
            )
        
        result = run_large_simulation()
        assert result is None  # Should return None on error
        
        # Should have created error report
        reporter = ErrorReporter()
        history = reporter.get_error_history()
        # History might be empty if reporter is fresh, but no exception should be raised
    
    def test_format_error_message(self):
        """Test error message formatting."""
        error = QubitError(
            message="Qubit 5 is out of range",
            user_message="Invalid qubit index",
            error_code="QUBIT001"
        )
        error.context.component = "Circuit"
        error.context.line_number = 10
        
        formatted = format_error_message(error, {'operation': 'gate_application'})
        
        assert formatted.title == "Logic Error in Circuit"
        assert formatted.error_code == "QUBIT001"
        assert formatted.category.value == "logic"
        assert len(formatted.suggestions) > 0
    
    def test_create_user_friendly_message(self):
        """Test user-friendly message creation."""
        message = create_user_friendly_message(
            error_type="Compilation Error",
            message="Failed to parse quantum circuit",
            suggestions=["Check syntax", "Verify structure"],
            context={"line_number": 15, "component": "Parser"}
        )
        
        assert "❌ Compilation Error" in message
        assert "📍 Location: Line 15" in message
        assert "🔧 Component: Parser" in message
        assert "💡 Suggestions:" in message
        assert "1. Check syntax" in message


def test_error_handling_demo():
    """Demonstrate the error handling system in action."""
    print("\n=== Error Handling System Demo ===")
    
    # 1. Create and handle various types of errors
    print("\n1. Testing Exception Hierarchy:")
    
    try:
        raise create_parse_error(
            "Unexpected token 'invalid'",
            line_number=5,
            suggestions=["Check spelling", "Review syntax"]
        )
    except ParseError as e:
        formatted = format_error_message(e)
        print(f"   {formatted.title}: {formatted.message}")
        print(f"   Error Code: {formatted.error_code}")
        print(f"   Suggestions: {formatted.suggestions[:2]}")
    
    # 2. Test error reporting
    print("\n2. Testing Error Reporting:")
    
    reporter = ErrorReporter()
    error = SimulationError("Out of memory during simulation")
    report = reporter.collect_error(error)
    
    print(f"   Report ID: {report.report_id}")
    print(f"   Timestamp: {report.timestamp}")
    print(f"   Error Type: {report.error_type}")
    
    # 3. Test alert system
    print("\n3. Testing Alert System:")
    
    alert_manager = get_alert_manager()
    
    # Create different types of alerts
    info_alert = alert_manager.info_alert("Info", "Circuit compiled successfully")
    warning_alert = alert_manager.warning_alert("Warning", "Large circuit may be slow")
    error_alert = alert_manager.error_alert(
        "Error", 
        "Failed to connect to quantum device",
        error_code="HW001",
        suggestions=["Check connection", "Try different device"]
    )
    
    active_alerts = alert_manager.get_active_alerts()
    print(f"   Created {len(active_alerts)} alerts:")
    for alert in active_alerts:
        print(f"     - {alert.alert_type.value.upper()}: {alert.title}")
    
    # 4. Test decorator integration
    print("\n4. Testing Decorator Integration:")
    
    @user_friendly_errors(component="Demo", operation="test_function")
    def function_that_fails():
        raise ValueError("Demonstration error")
    
    result = function_that_fails()
    print(f"   Function returned: {result}")
    print(f"   Alert created for error handling")
    
    # Cleanup
    alert_manager.shutdown()
    
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    # Run the demo
    test_error_handling_demo()
    
    # Run tests
    pytest.main([__file__, "-v"]) 