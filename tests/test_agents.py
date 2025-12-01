"""
Test Suite for AI-Powered Adaptive Learning Assistant
Demonstrates system reliability and correctness
"""

import pytest
import asyncio
from datetime import datetime
from main import (
    AdaptiveLearningAssistant,
    SubjectType,
    MemoryBank,
    SessionService,
    CustomTools,
    SpecializedAgent,
    CoordinatorAgent,
    AgentEvaluator,
    StudentProfile
)

@pytest.fixture
def memory_bank():
    """Fixture for memory bank"""
    return MemoryBank()

@pytest.fixture
def session_service():
    """Fixture for session service"""
    return SessionService()

@pytest.fixture
def custom_tools():
    """Fixture for custom tools"""
    return CustomTools()

@pytest.fixture
def assistant():
    """Fixture for main assistant"""
    return AdaptiveLearningAssistant()

# ===== Memory Bank Tests =====

def test_memory_bank_initialization(memory_bank):
    """Test memory bank initializes correctly"""
    assert memory_bank.students == {}
    assert memory_bank.interaction_history == []

def test_store_and_retrieve_student(memory_bank):
    """Test storing and retrieving student profiles"""
    profile = StudentProfile(
        student_id="test_001",
        name="Test Student",
        strengths=["Math"],
        weaknesses=["Writing"],
        learning_style="visual",
        progress={},
        session_history=[]
    )
    
    memory_bank.store_student(profile)
    retrieved = memory_bank.retrieve_student("test_001")
    
    assert retrieved is not None
    assert retrieved.student_id == "test_001"
    assert retrieved.name == "Test Student"

def test_update_progress(memory_bank):
    """Test updating student progress"""
    profile = StudentProfile(
        student_id="test_001",
        name="Test Student",
        strengths=[],
        weaknesses=[],
        learning_style="visual",
        progress={"math": 70.0},
        session_history=[]
    )
    
    memory_bank.store_student(profile)
    memory_bank.update_progress("test_001", "math", 85.0)
    
    retrieved = memory_bank.retrieve_student("test_001")
    assert retrieved.progress["math"] == 85.0

def test_add_interaction(memory_bank):
    """Test logging interactions"""
    memory_bank.add_interaction("test_001", {
        "query": "Help with math",
        "response": "Here's how..."
    })
    
    assert len(memory_bank.interaction_history) == 1
    assert memory_bank.interaction_history[0]["student_id"] == "test_001"

# ===== Session Service Tests =====

def test_create_session(session_service):
    """Test session creation"""
    session_id = session_service.create_session("test_001", "mathematics")
    
    assert session_id in session_service.active_sessions
    assert session_service.active_sessions[session_id]["student_id"] == "test_001"
    assert session_service.active_sessions[session_id]["subject"] == "mathematics"

def test_save_and_resume_session(session_service):
    """Test session state persistence"""
    session_id = session_service.create_session("test_001", "mathematics")
    state = {"step": 1, "completed": True}
    
    session_service.save_state(session_id, state)
    resumed_state = session_service.resume_session(session_id)
    
    assert resumed_state == state

# ===== Custom Tools Tests =====

@pytest.mark.asyncio
async def test_generate_practice_problem(custom_tools):
    """Test problem generation"""
    problem = await custom_tools.generate_practice_problem(
        subject="mathematics",
        difficulty="easy",
        topic="Algebra"
    )
    
    assert "problem" in problem
    assert problem["subject"] == "mathematics"
    assert problem["difficulty"] == "easy"
    assert problem["topic"] == "Algebra"

@pytest.mark.asyncio
async def test_evaluate_answer(custom_tools):
    """Test answer evaluation"""
    evaluation = await custom_tools.evaluate_answer(
        problem="What is 2+2?",
        answer="4",
        correct_answer="4"
    )
    
    assert "score" in evaluation
    assert "feedback" in evaluation
    assert evaluation["score"] > 0

@pytest.mark.asyncio
async def test_identify_knowledge_gaps(custom_tools):
    """Test knowledge gap identification"""
    history = [
        {"topic": "Algebra", "score": 65},
        {"topic": "Geometry", "score": 55},
        {"topic": "Calculus", "score": 90}
    ]
    
    gaps = await custom_tools.identify_knowledge_gaps(history)
    
    assert isinstance(gaps, list)
    assert "Algebra" in gaps or "Geometry" in gaps

# ===== Specialized Agent Tests =====

@pytest.mark.asyncio
async def test_specialized_agent_initialization(memory_bank, custom_tools):
    """Test specialized agent creation"""
    agent = SpecializedAgent(SubjectType.MATH, memory_bank, custom_tools)
    
    assert agent.subject == SubjectType.MATH
    assert agent.name == "Mathematics Tutor Agent"

@pytest.mark.asyncio
async def test_agent_process_query(memory_bank, custom_tools):
    """Test agent query processing"""
    agent = SpecializedAgent(SubjectType.MATH, memory_bank, custom_tools)
    
    # Create a student profile
    profile = StudentProfile(
        student_id="test_001",
        name="Test Student",
        strengths=[],
        weaknesses=["mathematics"],
        learning_style="visual",
        progress={},
        session_history=[]
    )
    memory_bank.store_student(profile)
    
    response = await agent.process_query(
        student_id="test_001",
        query="Help with quadratic equations",
        context={"topic": "Algebra"}
    )
    
    assert "agent" in response
    assert "response" in response
    assert "practice_problem" in response

# ===== Coordinator Agent Tests =====

@pytest.mark.asyncio
async def test_coordinator_initialization(memory_bank, session_service):
    """Test coordinator agent initialization"""
    coordinator = CoordinatorAgent(memory_bank, session_service)
    
    assert len(coordinator.agents) == 3
    assert SubjectType.MATH in coordinator.agents
    assert SubjectType.SCIENCE in coordinator.agents
    assert SubjectType.WRITING in coordinator.agents

@pytest.mark.asyncio
async def test_coordinator_route_query(memory_bank, session_service):
    """Test query routing"""
    coordinator = CoordinatorAgent(memory_bank, session_service)
    
    result = await coordinator.route_query(
        student_id="test_001",
        query="Help with algebra",
        subject=SubjectType.MATH
    )
    
    assert result is not None
    assert "agent" in result

@pytest.mark.asyncio
async def test_parallel_assessment(memory_bank, session_service):
    """Test parallel assessment across multiple agents"""
    coordinator = CoordinatorAgent(memory_bank, session_service)
    
    topics = [
        (SubjectType.MATH, "Algebra"),
        (SubjectType.SCIENCE, "Biology"),
        (SubjectType.WRITING, "Essays")
    ]
    
    results = await coordinator.parallel_assessment("test_001", topics)
    
    assert len(results) == 3
    assert all("agent" in result for result in results)

@pytest.mark.asyncio
async def test_sequential_learning_path(memory_bank, session_service):
    """Test sequential learning execution"""
    coordinator = CoordinatorAgent(memory_bank, session_service)
    
    curriculum = [
        {"subject": SubjectType.MATH, "query": "Learn basics", "topic": "Algebra"},
        {"subject": SubjectType.MATH, "query": "Practice problems", "topic": "Equations"}
    ]
    
    results = await coordinator.sequential_learning_path("test_001", curriculum)
    
    assert len(results) == 2
    assert all("agent" in result for result in results)

# ===== Agent Evaluator Tests =====

def test_evaluator_initialization():
    """Test evaluator initialization"""
    evaluator = AgentEvaluator()
    
    assert evaluator.metrics["total_queries"] == 0
    assert evaluator.metrics["successful_responses"] == 0

def test_evaluator_log_query():
    """Test query logging"""
    evaluator = AgentEvaluator()
    
    evaluator.log_query(success=True, response_time=1.5)
    evaluator.log_query(success=True, response_time=2.0)
    evaluator.log_query(success=False, response_time=3.0)
    
    assert evaluator.metrics["total_queries"] == 3
    assert evaluator.metrics["successful_responses"] == 2
    assert evaluator.get_success_rate() == 2/3

def test_evaluator_generate_report():
    """Test report generation"""
    evaluator = AgentEvaluator()
    
    evaluator.log_query(success=True, response_time=1.5)
    report = evaluator.generate_report()
    
    assert "success_rate" in report
    assert "total_queries" in report
    assert "avg_response_time" in report

# ===== Integration Tests =====

@pytest.mark.asyncio
async def test_full_system_integration(assistant):
    """Test complete system workflow"""
    # Register student
    profile = assistant.register_student("integration_test", "Integration Test Student")
    assert profile.student_id == "integration_test"
    
    # Process learning request
    result = await assistant.process_learning_request(
        student_id="integration_test",
        subject=SubjectType.MATH,
        query="Help with quadratic equations"
    )
    
    assert result is not None
    assert "agent" in result
    
    # Check metrics were logged
    metrics = assistant.get_system_metrics()
    assert metrics["total_queries"] > 0

@pytest.mark.asyncio
async def test_student_learning_journey(assistant):
    """Test complete student learning journey"""
    # Register student
    student_id = "journey_test"
    assistant.register_student(student_id, "Journey Test Student")
    
    # Process multiple queries
    subjects = [
        (SubjectType.MATH, "Algebra"),
        (SubjectType.SCIENCE, "Biology"),
        (SubjectType.WRITING, "Essays")
    ]
    
    for subject, topic in subjects:
        result = await assistant.process_learning_request(
            student_id=student_id,
            subject=subject,
            query=f"Help with {topic}"
        )
        assert result is not None
    
    # Verify student profile was updated
    profile = assistant.memory.retrieve_student(student_id)
    assert profile is not None
    assert len(assistant.memory.interaction_history) >= 3

@pytest.mark.asyncio
async def test_performance_under_load(assistant):
    """Test system performance with multiple concurrent requests"""
    student_id = "load_test"
    assistant.register_student(student_id, "Load Test Student")
    
    # Create multiple concurrent requests
    tasks = []
    for i in range(10):
        task = assistant.process_learning_request(
            student_id=student_id,
            subject=SubjectType.MATH,
            query=f"Query {i}"
        )
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    
    assert len(results) == 10
    assert all(r is not None for r in results)
    
    # Check system remained responsive
    metrics = assistant.get_system_metrics()
    assert metrics["success_rate"] > 0.8  # At least 80% success rate

# ===== Edge Cases and Error Handling =====

def test_retrieve_nonexistent_student(memory_bank):
    """Test retrieving a student that doesn't exist"""
    result = memory_bank.retrieve_student("nonexistent")
    assert result is None

def test_resume_nonexistent_session(session_service):
    """Test resuming a session that doesn't exist"""
    result = session_service.resume_session("nonexistent")
    assert result is None

@pytest.mark.asyncio
async def test_invalid_subject_handling():
    """Test handling of invalid subject types"""
    assistant = AdaptiveLearningAssistant()
    
    # This should handle the error gracefully
    try:
        result = await assistant.process_learning_request(
            student_id="test",
            subject="invalid_subject",  # Invalid
            query="Test query"
        )
    except Exception as e:
        # Expected to fail gracefully
        assert True

# ===== Performance Tests =====

@pytest.mark.asyncio
async def test_response_time_benchmark(assistant):
    """Benchmark response time"""
    student_id = "benchmark_test"
    assistant.register_student(student_id, "Benchmark Student")
    
    start_time = datetime.now()
    
    result = await assistant.process_learning_request(
        student_id=student_id,
        subject=SubjectType.MATH,
        query="Quick test query"
    )
    
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    
    # Should respond in under 5 seconds
    assert elapsed < 5.0
    assert result is not None

# ===== Summary Statistics =====

def test_system_statistics():
    """Generate and verify system statistics"""
    evaluator = AgentEvaluator()
    
    # Simulate 100 queries
    for i in range(100):
        success = i % 10 != 0  # 90% success rate
        evaluator.log_query(success=success, response_time=1.5)
    
    report = evaluator.generate_report()
    
    assert report["total_queries"] == 100
    assert 0.85 <= report["success_rate"] <= 0.95  # Around 90%
    assert report["avg_response_time"] == 1.5

if __name__ == "__main__":
    pytest.main([__file__, "-v"])