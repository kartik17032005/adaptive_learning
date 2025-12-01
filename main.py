"""
AI-Powered Adaptive Learning Assistant
Agents for Good - Capstone Project
Multi-Agent System with Memory, Tools, and Observability
"""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import json
import logging
from enum import Enum

# Configure logging for observability
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SubjectType(Enum):
    MATH = "mathematics"
    SCIENCE = "science"
    WRITING = "writing"

@dataclass
class StudentProfile:
    """Student profile with learning history"""
    student_id: str
    name: str
    strengths: List[str]
    weaknesses: List[str]
    learning_style: str
    progress: Dict[str, float]
    session_history: List[Dict[str, Any]]
    
class MemoryBank:
    """Long-term memory system for student data"""
    
    def __init__(self):
        self.students: Dict[str, StudentProfile] = {}
        self.interaction_history: List[Dict] = []
        logger.info("MemoryBank initialized")
    
    def store_student(self, profile: StudentProfile):
        """Store student profile"""
        self.students[profile.student_id] = profile
        logger.info(f"Stored profile for student: {profile.student_id}")
    
    def retrieve_student(self, student_id: str) -> Optional[StudentProfile]:
        """Retrieve student profile"""
        profile = self.students.get(student_id)
        if profile:
            logger.info(f"Retrieved profile for student: {student_id}")
        return profile
    
    def update_progress(self, student_id: str, subject: str, score: float):
        """Update student progress"""
        if student_id in self.students:
            self.students[student_id].progress[subject] = score
            logger.info(f"Updated progress for {student_id} in {subject}: {score}")
    
    def add_interaction(self, student_id: str, interaction: Dict):
        """Log interaction for observability"""
        interaction['timestamp'] = datetime.now().isoformat()
        interaction['student_id'] = student_id
        self.interaction_history.append(interaction)
        logger.info(f"Logged interaction for {student_id}")

class SessionService:
    """Manages student learning sessions"""
    
    def __init__(self):
        self.active_sessions: Dict[str, Dict] = {}
        self.session_state: Dict[str, Any] = {}
        logger.info("SessionService initialized")
    
    def create_session(self, student_id: str, subject: str) -> str:
        """Create new learning session"""
        session_id = f"session_{student_id}_{datetime.now().timestamp()}"
        self.active_sessions[session_id] = {
            'student_id': student_id,
            'subject': subject,
            'start_time': datetime.now().isoformat(),
            'status': 'active'
        }
        logger.info(f"Created session: {session_id}")
        return session_id
    
    def save_state(self, session_id: str, state: Dict):
        """Save session state for pause/resume"""
        self.session_state[session_id] = state
        logger.info(f"Saved state for session: {session_id}")
    
    def resume_session(self, session_id: str) -> Optional[Dict]:
        """Resume paused session"""
        state = self.session_state.get(session_id)
        if state:
            logger.info(f"Resumed session: {session_id}")
        return state

class CustomTools:
    """Custom tools for the learning assistant"""
    
    @staticmethod
    async def generate_practice_problem(subject: str, difficulty: str, topic: str) -> Dict:
        """Generate practice problems based on subject and difficulty"""
        logger.info(f"Generating {difficulty} problem for {subject} - {topic}")
        
        problems = {
            'mathematics': {
                'easy': f"Solve: 2x + 5 = 15 (Topic: {topic})",
                'medium': f"Find the derivative of f(x) = 3x² + 2x - 1 (Topic: {topic})",
                'hard': f"Prove by induction: 1+2+3+...+n = n(n+1)/2 (Topic: {topic})"
            },
            'science': {
                'easy': f"What is photosynthesis? (Topic: {topic})",
                'medium': f"Explain the process of cellular respiration (Topic: {topic})",
                'hard': f"Describe quantum entanglement and its implications (Topic: {topic})"
            },
            'writing': {
                'easy': f"Write a paragraph about your favorite hobby (Topic: {topic})",
                'medium': f"Compose a persuasive essay outline on climate change (Topic: {topic})",
                'hard': f"Analyze the symbolism in Shakespeare's Macbeth (Topic: {topic})"
            }
        }
        
        problem = problems.get(subject, {}).get(difficulty, "Problem generation failed")
        return {
            'problem': problem,
            'subject': subject,
            'difficulty': difficulty,
            'topic': topic,
            'generated_at': datetime.now().isoformat()
        }
    
    @staticmethod
    async def evaluate_answer(problem: str, answer: str, correct_answer: str) -> Dict:
        """Evaluate student's answer"""
        logger.info("Evaluating student answer")
        
        # Simplified evaluation logic
        score = 85 if answer.lower() in correct_answer.lower() else 60
        
        return {
            'score': score,
            'feedback': "Great job! Your understanding is solid." if score > 70 else "Let's review this concept together.",
            'evaluated_at': datetime.now().isoformat()
        }
    
    @staticmethod
    async def identify_knowledge_gaps(student_history: List[Dict]) -> List[str]:
        """Analyze student history to identify knowledge gaps"""
        logger.info("Analyzing knowledge gaps")
        
        gaps = []
        for session in student_history[-5:]:  # Last 5 sessions
            if session.get('score', 100) < 70:
                gaps.append(session.get('topic', 'Unknown topic'))
        
        return list(set(gaps))

class SpecializedAgent:
    """Base class for specialized subject agents"""
    
    def __init__(self, subject: SubjectType, memory: MemoryBank, tools: CustomTools):
        self.subject = subject
        self.memory = memory
        self.tools = tools
        self.name = f"{subject.value.title()} Tutor Agent"
        logger.info(f"Initialized {self.name}")
    
    async def process_query(self, student_id: str, query: str, context: Dict) -> Dict:
        """Process student query"""
        logger.info(f"{self.name} processing query for student {student_id}")
        
        # Retrieve student profile from memory
        profile = self.memory.retrieve_student(student_id)
        
        # Generate response based on student's level
        response = await self._generate_adaptive_response(query, profile, context)
        
        # Log interaction
        self.memory.add_interaction(student_id, {
            'agent': self.name,
            'query': query,
            'response': response
        })
        
        return response
    
    async def _generate_adaptive_response(self, query: str, profile: Optional[StudentProfile], context: Dict) -> Dict:
        """Generate response adapted to student's level"""
        
        if profile:
            difficulty = 'easy' if self.subject.value in profile.weaknesses else 'medium'
        else:
            difficulty = 'medium'
        
        # Use custom tools to generate practice problems
        problem = await self.tools.generate_practice_problem(
            self.subject.value,
            difficulty,
            context.get('topic', 'General')
        )
        
        return {
            'agent': self.name,
            'response': f"Let me help you with {query}",
            'practice_problem': problem,
            'personalized': profile is not None,
            'timestamp': datetime.now().isoformat()
        }

class CoordinatorAgent:
    """Coordinator agent that manages specialized agents"""
    
    def __init__(self, memory: MemoryBank, session_service: SessionService):
        self.memory = memory
        self.session_service = session_service
        self.tools = CustomTools()
        
        # Initialize specialized agents
        self.agents = {
            SubjectType.MATH: SpecializedAgent(SubjectType.MATH, memory, self.tools),
            SubjectType.SCIENCE: SpecializedAgent(SubjectType.SCIENCE, memory, self.tools),
            SubjectType.WRITING: SpecializedAgent(SubjectType.WRITING, memory, self.tools)
        }
        
        logger.info("CoordinatorAgent initialized with 3 specialized agents")
    
    async def route_query(self, student_id: str, query: str, subject: SubjectType) -> Dict:
        """Route query to appropriate specialized agent"""
        logger.info(f"Routing query to {subject.value} agent")
        
        # Create or resume session
        session_id = self.session_service.create_session(student_id, subject.value)
        
        # Get appropriate agent
        agent = self.agents.get(subject)
        
        if not agent:
            logger.error(f"No agent found for subject: {subject}")
            return {'error': 'Subject not supported'}
        
        # Process query through specialized agent
        context = {'session_id': session_id}
        response = await agent.process_query(student_id, query, context)
        
        return response
    
    async def parallel_assessment(self, student_id: str, topics: List[tuple]) -> List[Dict]:
        """Run parallel assessments across multiple agents"""
        logger.info(f"Running parallel assessment for student {student_id}")
        
        tasks = []
        for subject, topic in topics:
            agent = self.agents.get(subject)
            if agent:
                task = agent.process_query(student_id, f"Assess {topic}", {'topic': topic})
                tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        logger.info(f"Parallel assessment completed: {len(results)} results")
        
        return results
    
    async def sequential_learning_path(self, student_id: str, curriculum: List[Dict]) -> List[Dict]:
        """Execute sequential learning path"""
        logger.info(f"Starting sequential learning path for {student_id}")
        
        results = []
        for step in curriculum:
            subject = step['subject']
            query = step['query']
            
            agent = self.agents.get(subject)
            if agent:
                result = await agent.process_query(student_id, query, step)
                results.append(result)
                
                # Save state for pause/resume capability
                session_id = step.get('session_id')
                if session_id:
                    self.session_service.save_state(session_id, {
                        'completed_steps': len(results),
                        'current_step': step
                    })
        
        logger.info(f"Sequential learning path completed: {len(results)} steps")
        return results

class AgentEvaluator:
    """Evaluation system for agent performance"""
    
    def __init__(self):
        self.metrics = {
            'total_queries': 0,
            'successful_responses': 0,
            'average_response_time': 0,
            'student_satisfaction': []
        }
        logger.info("AgentEvaluator initialized")
    
    def log_query(self, success: bool, response_time: float):
        """Log query metrics"""
        self.metrics['total_queries'] += 1
        if success:
            self.metrics['successful_responses'] += 1
        
        # Update average response time
        current_avg = self.metrics['average_response_time']
        total = self.metrics['total_queries']
        self.metrics['average_response_time'] = (current_avg * (total - 1) + response_time) / total
        
        logger.info(f"Query logged. Success rate: {self.get_success_rate():.2%}")
    
    def get_success_rate(self) -> float:
        """Calculate success rate"""
        if self.metrics['total_queries'] == 0:
            return 0.0
        return self.metrics['successful_responses'] / self.metrics['total_queries']
    
    def generate_report(self) -> Dict:
        """Generate evaluation report"""
        report = {
            'success_rate': self.get_success_rate(),
            'total_queries': self.metrics['total_queries'],
            'avg_response_time': self.metrics['average_response_time'],
            'generated_at': datetime.now().isoformat()
        }
        logger.info("Generated evaluation report")
        return report

# Main Application
class AdaptiveLearningAssistant:
    """Main application orchestrating all components"""
    
    def __init__(self):
        self.memory = MemoryBank()
        self.session_service = SessionService()
        self.coordinator = CoordinatorAgent(self.memory, self.session_service)
        self.evaluator = AgentEvaluator()
        
        logger.info("AdaptiveLearningAssistant initialized")
    
    def register_student(self, student_id: str, name: str) -> StudentProfile:
        """Register new student"""
        profile = StudentProfile(
            student_id=student_id,
            name=name,
            strengths=[],
            weaknesses=[],
            learning_style="visual",
            progress={},
            session_history=[]
        )
        self.memory.store_student(profile)
        logger.info(f"Registered new student: {name}")
        return profile
    
    async def process_learning_request(self, student_id: str, subject: SubjectType, query: str) -> Dict:
        """Process student learning request"""
        start_time = datetime.now()
        
        try:
            result = await self.coordinator.route_query(student_id, query, subject)
            
            # Log evaluation metrics
            elapsed = (datetime.now() - start_time).total_seconds()
            self.evaluator.log_query(success=True, response_time=elapsed)
            
            return result
        
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            elapsed = (datetime.now() - start_time).total_seconds()
            self.evaluator.log_query(success=False, response_time=elapsed)
            return {'error': str(e)}
    
    def get_system_metrics(self) -> Dict:
        """Get system performance metrics"""
        return self.evaluator.generate_report()

# Demo Usage
async def main():
    """Demo the system"""
    print("=" * 60)
    print("AI-Powered Adaptive Learning Assistant")
    print("Agents for Good - Capstone Project")
    print("=" * 60)
    
    # Initialize system
    assistant = AdaptiveLearningAssistant()
    
    # Register student
    student = assistant.register_student("student_001", "Alice Johnson")
    print(f"\n✅ Registered student: {student.name}")
    
    # Process learning requests
    print("\n📚 Processing learning requests...")
    
    # Single query
    result1 = await assistant.process_learning_request(
        "student_001",
        SubjectType.MATH,
        "I need help with quadratic equations"
    )
    print(f"\n🎯 Math Query Result:")
    print(json.dumps(result1, indent=2, default=str))
    
    # Parallel assessment
    print("\n🔄 Running parallel assessment across subjects...")
    parallel_results = await assistant.coordinator.parallel_assessment(
        "student_001",
        [
            (SubjectType.MATH, "Algebra"),
            (SubjectType.SCIENCE, "Biology"),
            (SubjectType.WRITING, "Essays")
        ]
    )
    print(f"✅ Completed {len(parallel_results)} parallel assessments")
    
    # Sequential learning path
    print("\n📖 Starting sequential learning path...")
    curriculum = [
        {'subject': SubjectType.MATH, 'query': 'Learn basic algebra', 'topic': 'Algebra Basics'},
        {'subject': SubjectType.MATH, 'query': 'Practice quadratic equations', 'topic': 'Quadratics'}
    ]
    seq_results = await assistant.coordinator.sequential_learning_path("student_001", curriculum)
    print(f"✅ Completed {len(seq_results)} sequential steps")
    
    # Get system metrics
    print("\n📊 System Performance Metrics:")
    metrics = assistant.get_system_metrics()
    print(json.dumps(metrics, indent=2))
    
    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())