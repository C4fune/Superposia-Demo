"""
Community Features for Quantum Platform Marketplace

This module provides community-driven features including user profiles,
algorithm contributions, ratings, and discussion threads.
"""

import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from quantum_platform.observability.logging import get_logger


class ContributionType(Enum):
    """Types of community contributions."""
    ALGORITHM = "algorithm"
    PLUGIN = "plugin"
    TUTORIAL = "tutorial"
    EXAMPLE = "example"
    DOCUMENTATION = "documentation"
    BUG_REPORT = "bug_report"
    FEATURE_REQUEST = "feature_request"


@dataclass
class UserProfile:
    """User profile for community features."""
    user_id: str
    username: str
    display_name: str
    email: str
    bio: str = ""
    location: str = ""
    website: str = ""
    github_username: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    contributions: int = 0
    reputation: int = 0
    badges: List[str] = field(default_factory=list)
    favorite_algorithms: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "user_id": self.user_id,
            "username": self.username,
            "display_name": self.display_name,
            "email": self.email,
            "bio": self.bio,
            "location": self.location,
            "website": self.website,
            "github_username": self.github_username,
            "created_at": self.created_at.isoformat(),
            "contributions": self.contributions,
            "reputation": self.reputation,
            "badges": self.badges,
            "favorite_algorithms": self.favorite_algorithms
        }


@dataclass
class AlgorithmContribution:
    """Algorithm contribution from community."""
    contribution_id: str
    user_id: str
    algorithm_name: str
    description: str
    category: str
    tags: List[str]
    code_repository: str
    documentation_url: str = ""
    license: str = "MIT"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    downloads: int = 0
    rating: float = 0.0
    rating_count: int = 0
    status: str = "pending"  # pending, approved, rejected
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "contribution_id": self.contribution_id,
            "user_id": self.user_id,
            "algorithm_name": self.algorithm_name,
            "description": self.description,
            "category": self.category,
            "tags": self.tags,
            "code_repository": self.code_repository,
            "documentation_url": self.documentation_url,
            "license": self.license,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "downloads": self.downloads,
            "rating": self.rating,
            "rating_count": self.rating_count,
            "status": self.status
        }


@dataclass
class CommunityRating:
    """User rating for packages/algorithms."""
    rating_id: str
    user_id: str
    package_name: str
    rating: int  # 1-5
    review: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    helpful_votes: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rating_id": self.rating_id,
            "user_id": self.user_id,
            "package_name": self.package_name,
            "rating": self.rating,
            "review": self.review,
            "created_at": self.created_at.isoformat(),
            "helpful_votes": self.helpful_votes
        }


@dataclass
class DiscussionThread:
    """Discussion thread for community."""
    thread_id: str
    user_id: str
    title: str
    content: str
    category: str
    tags: List[str]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    reply_count: int = 0
    views: int = 0
    is_pinned: bool = False
    is_locked: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "thread_id": self.thread_id,
            "user_id": self.user_id,
            "title": self.title,
            "content": self.content,
            "category": self.category,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "reply_count": self.reply_count,
            "views": self.views,
            "is_pinned": self.is_pinned,
            "is_locked": self.is_locked
        }


class CommunityManager:
    """Manages community features and interactions."""
    
    def __init__(self, data_path: str = "community_data"):
        """Initialize community manager."""
        self.logger = get_logger("CommunityManager")
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        # Data storage
        self.users: Dict[str, UserProfile] = {}
        self.contributions: Dict[str, AlgorithmContribution] = {}
        self.ratings: Dict[str, CommunityRating] = {}
        self.discussions: Dict[str, DiscussionThread] = {}
        
        # Load existing data
        self._load_data()
        
        self.logger.info("Community manager initialized")
    
    def _load_data(self):
        """Load community data from storage."""
        try:
            # Load users
            users_file = self.data_path / "users.json"
            if users_file.exists():
                with open(users_file, 'r') as f:
                    users_data = json.load(f)
                    for user_data in users_data:
                        user_data['created_at'] = datetime.fromisoformat(user_data['created_at'])
                        user = UserProfile(**user_data)
                        self.users[user.user_id] = user
            
            # Load contributions
            contributions_file = self.data_path / "contributions.json"
            if contributions_file.exists():
                with open(contributions_file, 'r') as f:
                    contributions_data = json.load(f)
                    for contrib_data in contributions_data:
                        contrib_data['created_at'] = datetime.fromisoformat(contrib_data['created_at'])
                        contrib_data['updated_at'] = datetime.fromisoformat(contrib_data['updated_at'])
                        contrib = AlgorithmContribution(**contrib_data)
                        self.contributions[contrib.contribution_id] = contrib
            
            # Load ratings
            ratings_file = self.data_path / "ratings.json"
            if ratings_file.exists():
                with open(ratings_file, 'r') as f:
                    ratings_data = json.load(f)
                    for rating_data in ratings_data:
                        rating_data['created_at'] = datetime.fromisoformat(rating_data['created_at'])
                        rating = CommunityRating(**rating_data)
                        self.ratings[rating.rating_id] = rating
            
            # Load discussions
            discussions_file = self.data_path / "discussions.json"
            if discussions_file.exists():
                with open(discussions_file, 'r') as f:
                    discussions_data = json.load(f)
                    for discussion_data in discussions_data:
                        discussion_data['created_at'] = datetime.fromisoformat(discussion_data['created_at'])
                        discussion_data['updated_at'] = datetime.fromisoformat(discussion_data['updated_at'])
                        discussion = DiscussionThread(**discussion_data)
                        self.discussions[discussion.thread_id] = discussion
            
            self.logger.info(f"Loaded {len(self.users)} users, {len(self.contributions)} contributions")
            
        except Exception as e:
            self.logger.error(f"Failed to load community data: {e}")
    
    def _save_data(self):
        """Save community data to storage."""
        try:
            # Save users
            users_file = self.data_path / "users.json"
            with open(users_file, 'w') as f:
                json.dump([user.to_dict() for user in self.users.values()], f, indent=2)
            
            # Save contributions
            contributions_file = self.data_path / "contributions.json"
            with open(contributions_file, 'w') as f:
                json.dump([contrib.to_dict() for contrib in self.contributions.values()], f, indent=2)
            
            # Save ratings
            ratings_file = self.data_path / "ratings.json"
            with open(ratings_file, 'w') as f:
                json.dump([rating.to_dict() for rating in self.ratings.values()], f, indent=2)
            
            # Save discussions
            discussions_file = self.data_path / "discussions.json"
            with open(discussions_file, 'w') as f:
                json.dump([discussion.to_dict() for discussion in self.discussions.values()], f, indent=2)
            
        except Exception as e:
            self.logger.error(f"Failed to save community data: {e}")
    
    def create_user_profile(self, user_data: Dict[str, Any]) -> UserProfile:
        """Create a new user profile."""
        user = UserProfile(**user_data)
        self.users[user.user_id] = user
        self._save_data()
        self.logger.info(f"Created user profile for {user.username}")
        return user
    
    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile by ID."""
        return self.users.get(user_id)
    
    def submit_contribution(self, contrib_data: Dict[str, Any]) -> AlgorithmContribution:
        """Submit a new algorithm contribution."""
        contribution = AlgorithmContribution(**contrib_data)
        self.contributions[contribution.contribution_id] = contribution
        
        # Update user contributions count
        if contribution.user_id in self.users:
            self.users[contribution.user_id].contributions += 1
        
        self._save_data()
        self.logger.info(f"New contribution submitted: {contribution.algorithm_name}")
        return contribution
    
    def get_contributions(self, user_id: Optional[str] = None, 
                         status: Optional[str] = None) -> List[AlgorithmContribution]:
        """Get algorithm contributions."""
        contributions = list(self.contributions.values())
        
        if user_id:
            contributions = [c for c in contributions if c.user_id == user_id]
        
        if status:
            contributions = [c for c in contributions if c.status == status]
        
        return contributions
    
    def rate_package(self, rating_data: Dict[str, Any]) -> CommunityRating:
        """Rate a package."""
        rating = CommunityRating(**rating_data)
        self.ratings[rating.rating_id] = rating
        self._save_data()
        self.logger.info(f"Package {rating.package_name} rated by user {rating.user_id}")
        return rating
    
    def get_package_ratings(self, package_name: str) -> List[CommunityRating]:
        """Get all ratings for a package."""
        return [r for r in self.ratings.values() if r.package_name == package_name]
    
    def create_discussion_thread(self, thread_data: Dict[str, Any]) -> DiscussionThread:
        """Create a new discussion thread."""
        thread = DiscussionThread(**thread_data)
        self.discussions[thread.thread_id] = thread
        self._save_data()
        self.logger.info(f"New discussion thread created: {thread.title}")
        return thread
    
    def get_discussion_threads(self, category: Optional[str] = None) -> List[DiscussionThread]:
        """Get discussion threads."""
        threads = list(self.discussions.values())
        
        if category:
            threads = [t for t in threads if t.category == category]
        
        return sorted(threads, key=lambda t: t.updated_at, reverse=True)
    
    def get_community_stats(self) -> Dict[str, Any]:
        """Get community statistics."""
        return {
            "total_users": len(self.users),
            "total_contributions": len(self.contributions),
            "approved_contributions": len([c for c in self.contributions.values() if c.status == "approved"]),
            "total_ratings": len(self.ratings),
            "total_discussions": len(self.discussions),
            "top_contributors": self._get_top_contributors(),
            "popular_algorithms": self._get_popular_algorithms()
        }
    
    def _get_top_contributors(self) -> List[Dict[str, Any]]:
        """Get top contributors by reputation."""
        users = sorted(self.users.values(), key=lambda u: u.reputation, reverse=True)
        return [
            {
                "username": user.username,
                "contributions": user.contributions,
                "reputation": user.reputation
            }
            for user in users[:10]
        ]
    
    def _get_popular_algorithms(self) -> List[Dict[str, Any]]:
        """Get most popular algorithms."""
        contributions = sorted(self.contributions.values(), key=lambda c: c.downloads, reverse=True)
        return [
            {
                "algorithm_name": contrib.algorithm_name,
                "downloads": contrib.downloads,
                "rating": contrib.rating,
                "author": self.users.get(contrib.user_id, {}).get("username", "Unknown")
            }
            for contrib in contributions[:10]
        ] 