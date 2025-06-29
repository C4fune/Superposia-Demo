"""
Log Viewer and Analysis Tools

This module provides tools for viewing, filtering, and analyzing log files and
system monitoring data. It includes log parsing, filtering capabilities, and
analysis tools for troubleshooting and system understanding.
"""

import re
import json
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable, Iterator, Tuple
from quantum_platform.observability.logging import get_logger

@dataclass
class LogEntry:
    """Parsed log entry with structured data."""
    timestamp: datetime
    component: str
    level: str
    message: str
    raw_line: str
    line_number: int
    additional_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert log entry to dictionary format."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'component': self.component,
            'level': self.level,
            'message': self.message,
            'line_number': self.line_number,
            'additional_data': self.additional_data
        }

@dataclass
class LogFilter:
    """Configuration for filtering log entries."""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    components: Optional[List[str]] = None
    levels: Optional[List[str]] = None
    message_pattern: Optional[str] = None
    exclude_pattern: Optional[str] = None
    max_entries: Optional[int] = None
    
    def matches(self, entry: LogEntry) -> bool:
        """Check if a log entry matches this filter."""
        # Time range filter
        if self.start_time and entry.timestamp < self.start_time:
            return False
        if self.end_time and entry.timestamp > self.end_time:
            return False
        
        # Component filter
        if self.components and entry.component not in self.components:
            return False
        
        # Level filter
        if self.levels and entry.level not in self.levels:
            return False
        
        # Message pattern filter
        if self.message_pattern:
            if not re.search(self.message_pattern, entry.message, re.IGNORECASE):
                return False
        
        # Exclude pattern filter
        if self.exclude_pattern:
            if re.search(self.exclude_pattern, entry.message, re.IGNORECASE):
                return False
        
        return True

class LogAnalyzer:
    """
    Advanced log analysis and pattern detection.
    
    Provides statistical analysis, pattern detection, and insights
    into system behavior based on log data.
    """
    
    def __init__(self):
        """Initialize log analyzer."""
        self.logger = get_logger("LogAnalyzer")
    
    def analyze_log_patterns(self, entries: List[LogEntry]) -> Dict[str, Any]:
        """
        Analyze log entries for patterns and statistics.
        
        Args:
            entries: List of log entries to analyze
            
        Returns:
            Dictionary containing analysis results
        """
        if not entries:
            return {'error': 'No log entries provided'}
        
        # Basic statistics
        total_entries = len(entries)
        level_counts = Counter(entry.level for entry in entries)
        component_counts = Counter(entry.component for entry in entries)
        
        # Time analysis
        time_span = entries[-1].timestamp - entries[0].timestamp if len(entries) > 1 else timedelta(0)
        entries_per_hour = total_entries / max(time_span.total_seconds() / 3600, 1)
        
        # Error analysis
        error_entries = [e for e in entries if e.level in ['ERROR', 'CRITICAL']]
        error_patterns = self._analyze_error_patterns(error_entries)
        
        # Performance analysis
        performance_entries = [e for e in entries if 'completed' in e.message.lower() or 'duration' in e.message.lower()]
        performance_analysis = self._analyze_performance_logs(performance_entries)
        
        # Component activity analysis
        component_activity = self._analyze_component_activity(entries)
        
        return {
            'summary': {
                'total_entries': total_entries,
                'time_span_hours': time_span.total_seconds() / 3600,
                'entries_per_hour': entries_per_hour,
                'unique_components': len(component_counts),
                'error_rate': len(error_entries) / total_entries if total_entries > 0 else 0
            },
            'level_distribution': dict(level_counts),
            'component_distribution': dict(component_counts),
            'error_analysis': error_patterns,
            'performance_analysis': performance_analysis,
            'component_activity': component_activity,
            'analysis_timestamp': datetime.now().isoformat()
        }
    
    def _analyze_error_patterns(self, error_entries: List[LogEntry]) -> Dict[str, Any]:
        """Analyze error patterns in log entries."""
        if not error_entries:
            return {'total_errors': 0}
        
        # Group errors by message patterns
        error_groups = defaultdict(list)
        for entry in error_entries:
            # Simple pattern extraction - group by first few words
            pattern = ' '.join(entry.message.split()[:5])
            error_groups[pattern].append(entry)
        
        # Find most common errors
        common_errors = sorted(
            [(pattern, len(entries)) for pattern, entries in error_groups.items()],
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # Error timeline
        error_timeline = []
        for entry in error_entries[-20:]:  # Last 20 errors
            error_timeline.append({
                'timestamp': entry.timestamp.isoformat(),
                'component': entry.component,
                'message': entry.message[:100] + ('...' if len(entry.message) > 100 else '')
            })
        
        return {
            'total_errors': len(error_entries),
            'common_error_patterns': common_errors,
            'recent_errors': error_timeline,
            'components_with_errors': list(set(e.component for e in error_entries))
        }
    
    def _analyze_performance_logs(self, performance_entries: List[LogEntry]) -> Dict[str, Any]:
        """Analyze performance-related log entries."""
        if not performance_entries:
            return {'performance_entries': 0}
        
        # Extract duration information
        durations = []
        operations = defaultdict(list)
        
        for entry in performance_entries:
            # Try to extract duration from message
            duration_match = re.search(r'(\d+\.?\d*)\s*s', entry.message)
            if duration_match:
                duration = float(duration_match.group(1))
                durations.append(duration)
                
                # Try to extract operation name
                if 'completed' in entry.message.lower():
                    op_match = re.search(r'completed\s+(\w+)', entry.message, re.IGNORECASE)
                    if op_match:
                        operations[op_match.group(1)].append(duration)
        
        # Calculate statistics
        if durations:
            avg_duration = sum(durations) / len(durations)
            max_duration = max(durations)
            min_duration = min(durations)
        else:
            avg_duration = max_duration = min_duration = 0
        
        # Operation statistics
        operation_stats = {}
        for op, times in operations.items():
            if times:
                operation_stats[op] = {
                    'count': len(times),
                    'avg_duration': sum(times) / len(times),
                    'max_duration': max(times),
                    'min_duration': min(times)
                }
        
        return {
            'performance_entries': len(performance_entries),
            'total_operations': len(durations),
            'avg_duration': avg_duration,
            'max_duration': max_duration,
            'min_duration': min_duration,
            'operation_statistics': operation_stats
        }
    
    def _analyze_component_activity(self, entries: List[LogEntry]) -> Dict[str, Any]:
        """Analyze activity patterns by component."""
        component_timeline = defaultdict(list)
        component_levels = defaultdict(Counter)
        
        for entry in entries:
            component_timeline[entry.component].append(entry.timestamp)
            component_levels[entry.component][entry.level] += 1
        
        # Calculate activity metrics for each component
        activity_analysis = {}
        for component, timestamps in component_timeline.items():
            if len(timestamps) > 1:
                time_diffs = [
                    (timestamps[i] - timestamps[i-1]).total_seconds()
                    for i in range(1, len(timestamps))
                ]
                avg_interval = sum(time_diffs) / len(time_diffs)
            else:
                avg_interval = 0
            
            activity_analysis[component] = {
                'total_entries': len(timestamps),
                'avg_interval_seconds': avg_interval,
                'level_distribution': dict(component_levels[component]),
                'first_activity': timestamps[0].isoformat() if timestamps else None,
                'last_activity': timestamps[-1].isoformat() if timestamps else None
            }
        
        return activity_analysis
    
    def detect_anomalies(self, entries: List[LogEntry], 
                        time_window: timedelta = timedelta(minutes=5)) -> List[Dict[str, Any]]:
        """
        Detect potential anomalies in log patterns.
        
        Args:
            entries: List of log entries to analyze
            time_window: Time window for anomaly detection
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        # Group entries by time windows
        if not entries:
            return anomalies
        
        start_time = entries[0].timestamp
        end_time = entries[-1].timestamp
        current_time = start_time
        
        while current_time < end_time:
            window_end = current_time + time_window
            window_entries = [
                e for e in entries 
                if current_time <= e.timestamp < window_end
            ]
            
            if window_entries:
                window_analysis = self._analyze_time_window(window_entries, current_time, window_end)
                if window_analysis['is_anomalous']:
                    anomalies.append(window_analysis)
            
            current_time = window_end
        
        return anomalies
    
    def _analyze_time_window(self, entries: List[LogEntry], 
                           start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Analyze a specific time window for anomalies."""
        error_count = len([e for e in entries if e.level in ['ERROR', 'CRITICAL']])
        total_count = len(entries)
        error_rate = error_count / total_count if total_count > 0 else 0
        
        # Simple anomaly detection criteria
        is_anomalous = (
            error_rate > 0.1 or  # More than 10% errors
            error_count > 5 or   # More than 5 errors in window
            total_count > 100    # Unusually high activity
        )
        
        return {
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'total_entries': total_count,
            'error_count': error_count,
            'error_rate': error_rate,
            'is_anomalous': is_anomalous,
            'anomaly_reasons': [
                'High error rate' if error_rate > 0.1 else None,
                'High error count' if error_count > 5 else None,
                'High activity' if total_count > 100 else None
            ]
        }

class LogViewer:
    """
    Comprehensive log viewing and filtering interface.
    
    Provides capabilities to read, parse, filter, and display log files
    with various formatting and analysis options.
    """
    
    def __init__(self, log_file_path: Optional[str] = None):
        """
        Initialize log viewer.
        
        Args:
            log_file_path: Path to the log file to view
        """
        self.log_file_path = log_file_path or "logs/quantum_platform.log"
        self.logger = get_logger("LogViewer")
        self.analyzer = LogAnalyzer()
        
        # Regex pattern for parsing log entries
        self.log_pattern = re.compile(
            r'(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})\s+'
            r'\[(?P<component>[^\]]+)\]\s+'
            r'(?P<level>\w+):\s+'
            r'(?P<message>.*)'
        )
    
    def parse_log_file(self, file_path: Optional[str] = None) -> Iterator[LogEntry]:
        """
        Parse log file and yield log entries.
        
        Args:
            file_path: Path to log file, or None to use default
            
        Yields:
            LogEntry objects for each parsed line
        """
        path = Path(file_path or self.log_file_path)
        
        if not path.exists():
            self.logger.warning(f"Log file not found: {path}")
            return
        
        try:
            with open(path, 'r', encoding='utf-8') as file:
                for line_number, line in enumerate(file, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    entry = self._parse_log_line(line, line_number)
                    if entry:
                        yield entry
                        
        except Exception as e:
            self.logger.error(f"Error reading log file {path}: {e}")
    
    def _parse_log_line(self, line: str, line_number: int) -> Optional[LogEntry]:
        """Parse a single log line into a LogEntry."""
        match = self.log_pattern.match(line)
        if not match:
            # Handle lines that don't match the standard format
            return LogEntry(
                timestamp=datetime.now(),
                component="Unknown",
                level="INFO",
                message=line,
                raw_line=line,
                line_number=line_number
            )
        
        try:
            timestamp_str = match.group('timestamp')
            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f')
            
            return LogEntry(
                timestamp=timestamp,
                component=match.group('component'),
                level=match.group('level'),
                message=match.group('message'),
                raw_line=line,
                line_number=line_number
            )
        except ValueError as e:
            self.logger.warning(f"Error parsing timestamp in line {line_number}: {e}")
            return None
    
    def get_filtered_logs(self, log_filter: LogFilter, 
                         file_path: Optional[str] = None) -> List[LogEntry]:
        """
        Get filtered log entries.
        
        Args:
            log_filter: Filter configuration
            file_path: Path to log file
            
        Returns:
            List of filtered log entries
        """
        entries = []
        count = 0
        
        for entry in self.parse_log_file(file_path):
            if log_filter.matches(entry):
                entries.append(entry)
                count += 1
                
                if log_filter.max_entries and count >= log_filter.max_entries:
                    break
        
        return entries
    
    def tail_logs(self, lines: int = 50, follow: bool = False, 
                 log_filter: Optional[LogFilter] = None) -> Union[List[LogEntry], Iterator[LogEntry]]:
        """
        Get the last N lines from the log file.
        
        Args:
            lines: Number of lines to retrieve
            follow: Whether to continuously follow the log file
            log_filter: Optional filter to apply
            
        Returns:
            List of log entries or iterator if following
        """
        if follow:
            # Return an iterator for continuous following
            return self._follow_logs(log_filter)
        else:
            # Return last N entries
            all_entries = list(self.parse_log_file())
            filtered_entries = all_entries
            
            if log_filter:
                filtered_entries = [e for e in all_entries if log_filter.matches(e)]
            
            return filtered_entries[-lines:] if filtered_entries else []
    
    def _follow_logs(self, log_filter: Optional[LogFilter] = None) -> Iterator[LogEntry]:
        """Follow log file for new entries (like tail -f)."""
        import time
        
        path = Path(self.log_file_path)
        if not path.exists():
            return
        
        # Start from end of file
        with open(path, 'r', encoding='utf-8') as file:
            file.seek(0, 2)  # Go to end of file
            
            while True:
                line = file.readline()
                if line:
                    line = line.strip()
                    if line:
                        entry = self._parse_log_line(line, 0)
                        if entry and (not log_filter or log_filter.matches(entry)):
                            yield entry
                else:
                    time.sleep(0.1)  # Wait for new content
    
    def search_logs(self, search_term: str, context_lines: int = 2,
                   case_sensitive: bool = False) -> List[Dict[str, Any]]:
        """
        Search for specific terms in log files with context.
        
        Args:
            search_term: Term to search for
            context_lines: Number of context lines before/after match
            case_sensitive: Whether search is case sensitive
            
        Returns:
            List of search results with context
        """
        results = []
        all_entries = list(self.parse_log_file())
        
        flags = 0 if case_sensitive else re.IGNORECASE
        pattern = re.compile(re.escape(search_term), flags)
        
        for i, entry in enumerate(all_entries):
            if pattern.search(entry.message):
                # Get context entries
                context_start = max(0, i - context_lines)
                context_end = min(len(all_entries), i + context_lines + 1)
                context_entries = all_entries[context_start:context_end]
                
                results.append({
                    'match_entry': entry.to_dict(),
                    'match_index': i,
                    'context_entries': [e.to_dict() for e in context_entries],
                    'context_start_index': context_start
                })
        
        return results
    
    def export_filtered_logs(self, log_filter: LogFilter, 
                           output_file: str, format_type: str = "json"):
        """
        Export filtered logs to a file.
        
        Args:
            log_filter: Filter configuration
            output_file: Output file path
            format_type: Export format ("json", "csv", "txt")
        """
        entries = self.get_filtered_logs(log_filter)
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format_type == "json":
            with open(output_path, 'w', encoding='utf-8') as file:
                json.dump([e.to_dict() for e in entries], file, indent=2)
                
        elif format_type == "csv":
            import csv
            with open(output_path, 'w', newline='', encoding='utf-8') as file:
                if entries:
                    writer = csv.DictWriter(file, fieldnames=['timestamp', 'component', 'level', 'message'])
                    writer.writeheader()
                    for entry in entries:
                        writer.writerow({
                            'timestamp': entry.timestamp.isoformat(),
                            'component': entry.component,
                            'level': entry.level,
                            'message': entry.message
                        })
                        
        elif format_type == "txt":
            with open(output_path, 'w', encoding='utf-8') as file:
                for entry in entries:
                    file.write(f"{entry.raw_line}\n")
        
        self.logger.info(f"Exported {len(entries)} log entries to {output_path}")
    
    def get_log_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get a summary of recent log activity.
        
        Args:
            hours: Number of hours to analyze
            
        Returns:
            Dictionary with log summary information
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        log_filter = LogFilter(start_time=cutoff_time)
        entries = self.get_filtered_logs(log_filter)
        
        if not entries:
            return {'message': 'No recent log entries found'}
        
        # Basic analysis
        analysis = self.analyzer.analyze_log_patterns(entries)
        
        # Add recent errors
        recent_errors = [
            e.to_dict() for e in entries 
            if e.level in ['ERROR', 'CRITICAL']
        ][-10:]  # Last 10 errors
        
        return {
            'time_period_hours': hours,
            'analysis': analysis,
            'recent_errors': recent_errors,
            'summary_timestamp': datetime.now().isoformat()
        }

# Convenience functions
def create_filter(start_time: Optional[datetime] = None, 
                 end_time: Optional[datetime] = None,
                 components: Optional[List[str]] = None,
                 levels: Optional[List[str]] = None,
                 message_pattern: Optional[str] = None,
                 max_entries: Optional[int] = None) -> LogFilter:
    """Create a log filter with specified parameters."""
    return LogFilter(
        start_time=start_time,
        end_time=end_time,
        components=components,
        levels=levels,
        message_pattern=message_pattern,
        max_entries=max_entries
    )

def quick_analysis(log_file: Optional[str] = None, hours: int = 24) -> Dict[str, Any]:
    """Perform quick analysis of recent logs."""
    viewer = LogViewer(log_file)
    return viewer.get_log_summary(hours) 