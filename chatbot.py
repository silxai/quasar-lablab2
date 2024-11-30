import streamlit as st
import torch
import json
from datetime import datetime
import random
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import hashlib
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
from collections import Counter
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM
import together
from pathlib import Path
import json
import time
import asyncio
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Quasar Dataset Generator",
    page_icon="🌟",
    layout="wide"
)

def extract_keywords(text, max_keywords=5):
    # Simple keyword extraction based on word frequency
    words = text.lower().split()
    # Remove common words and short words
    words = [word for word in words if len(word) > 4]
    # Get most common words
    word_freq = pd.Series(words).value_counts()
    return word_freq.head(max_keywords).index.tolist()

def validate_entry(entry):
    checks = []
    
    # Check if response is not empty
    if not entry["response"].strip():
        checks.append("empty_response")
    
    # Check if required fields exist
    required_fields = ["prompt", "response", "metadata"]
    if not all(field in entry for field in required_fields):
        checks.append("missing_fields")
    
    return len(checks) == 0, checks

@dataclass
class QualityMetrics:
    complexity_score: float
    diversity_score: float
    technical_density: float
    avg_sentence_length: float
    word_count: int

class QualityTargets:
    def __init__(self):
        self.targets = {
            'complexity': {
                'min': 0.6,
                'target': 0.7,
                'max': 0.9
            },
            'diversity': {
                'min': 0.7,
                'target': 0.8,
                'max': 0.95
            },
            'technical_density': {
                'min': 0.1,
                'target': 0.15,
                'max': 0.3
            },
            'word_count': {
                'min': 50,
                'target': 100,
                'max': 200
            }
        }

class EnhancedDatasetGenerator:
    def __init__(
        self,
        model_name: str = "google/gemma-2-27b-it",
        temperature: float = 0.7,
        max_context_length: int = 2048
    ):
        st.write(f"🔄 Initializing with model: {model_name}")
        self.model_name = model_name
        self.temperature = temperature
        self.max_context_length = max_context_length
        self.context_data = None
        
        # Initialize Together AI
        together.api_key = "b2934a0d84d45a7511b9e1e9f62db2f6d2a7e388521f367ea1d2de47635ad201"  # Replace with your API key
        
        try:
            st.write("📚 Setting up generation...")
            self.model = model_name
            
            # Initialize the embedding model
            st.write("🔍 Loading embedding model...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            st.write("✅ Embedding model loaded successfully")
            
            st.write("✅ Setup completed successfully")
            
        except Exception as e:
            st.error(f"❌ Error in initialization: {str(e)}")
            raise

        self.quality_targets = QualityTargets()

    def add_context_data(self, source_files):
        """Process uploaded files into context knowledge"""
        st.write("📝 Processing uploaded files...")
        context_data = []
        for file in source_files:
            st.write(f"  - Reading {file.name}")
            content = file.read().decode('utf-8')
            if file.name.endswith('.txt'):
                lines = content.split('\n')
                st.write(f"    Found {len(lines)} lines in txt file")
                context_data.extend(lines)
            elif file.name.endswith('.csv'):
                df = pd.read_csv(pd.StringIO(content))
                st.write(f"    Found {len(df)} rows in csv file")
                context_data.extend(df.iloc[:, 0].tolist())
            elif file.name.endswith(('.json', '.jsonl')):
                lines = [line for line in content.split('\n') if line.strip()]
                st.write(f"    Found {len(lines)} lines in json file")
                for line in lines:
                    try:
                        data = json.loads(line)
                        if isinstance(data, dict) and 'text' in data:
                            context_data.append(data['text'])
                    except json.JSONDecodeError:
                        continue
        
        self.context_data = "\n".join(context_data)
        return len(context_data)

    def generate_dataset(self, topic: str, concepts: List[str], num_samples: int) -> List[Dict]:
        """Generate high-quality conversational dataset"""
        st.write(f"🎯 Generating enhanced dataset for topic: {topic}")
        dataset = []
        
        total_steps = len(concepts) * num_samples
        current_step = 0
        
        # Create a placeholder for the dashboard at the top
        dashboard_placeholder = st.empty()
        
        for concept in concepts:
            try:
                concept_context = self._get_relevant_context(concept)
                used_prompts = set()
                
                for i in range(num_samples):
                    try:
                        # Log prompt generation
                        st.write("      Generating unique prompt...")
                        attempt = 0
                        while attempt < 5:  # Limit attempts to avoid infinite loop
                            user_query = self._generate_unique_prompt(concept, topic)
                            if user_query not in used_prompts:
                                used_prompts.add(user_query)
                                st.write(f"      Generated prompt: {user_query}")
                                break
                            attempt += 1
                        
                        # Log response generation
                        st.write("      Generating response...")
                        response = self._generate_enhanced_response(
                            user_query=user_query,
                            concept=concept,
                            topic=topic,
                            context=concept_context
                        )
                        
                        if not response:
                            raise ValueError("Empty response generated")
                        
                        st.write(f"      Generated {len(response.split())} words")
                        
                        # Calculate metrics
                        st.write("      Calculating quality metrics...")
                        metrics = self._calculate_quality_metrics(response)
                        
                        # Add to dataset
                        cleaned_response = response.split("Assistant:")[-1].strip()
                        entry = {
                            "prompt": user_query,
                            "response": cleaned_response,
                            "metadata": {
                                "topic": topic,
                                "concept": concept,
                                "model": self.model_name,
                                "has_context": bool(self.context_data),
                                "type": "chat",
                                "quality_metrics": metrics.__dict__,
                                "generation_timestamp": datetime.now().isoformat()
                            }
                        }
                        dataset.append(entry)
                        current_step += 1
                        
                        # Update dashboard after each generation
                        with dashboard_placeholder:
                            self.create_dashboard(dataset, current_step, total_steps)
                        
                    except Exception as e:
                        st.error(f"❌ Error generating sample {i+1}: {str(e)}")
                        continue
                
            except Exception as e:
                st.error(f"❌ Error processing concept {concept}: {str(e)}")
                continue
        
        if not dataset:
            st.error("❌ No samples were generated successfully")
            raise ValueError("Failed to generate any valid samples")
        
        st.success(f"✅ Generated {len(dataset)} samples successfully")
        return dataset

    def _get_relevant_context(self, concept: str) -> str:
        """Get most relevant context using semantic search"""
        if not self.context_data:
            return ""
            
        # Split context into chunks
        context_chunks = self.context_data.split('\n')
        
        # Get embeddings
        concept_embedding = self.embedding_model.encode([concept])[0]
        chunk_embeddings = self.embedding_model.encode(context_chunks)
        
        # Calculate similarities
        similarities = cosine_similarity([concept_embedding], chunk_embeddings)[0]
        
        # Get top 3 most relevant chunks
        top_indices = similarities.argsort()[-3:][::-1]
        relevant_chunks = [context_chunks[i] for i in top_indices]
        
        return "\n".join(relevant_chunks)

    def _generate_enhanced_response(self, user_query: str, concept: str, topic: str, context: str) -> str:
        """Generate response using Together AI API"""
        try:
            messages = [
                {
                    "role": "system",
                    "content": f"You are an expert in {topic}, specifically about {concept}. Use this context in your response: {context}"
                },
                {
                    "role": "user",
                    "content": user_query
                }
            ]

            st.write("        Generating response via API...")
            
            # Initialize Together with API key
            together.api_key = "b2934a0d84d45a7511b9e1e9f62db2f6d2a7e388521f367ea1d2de47635ad201"
            client = together.Together(
                api_key="b2934a0d84d45a7511b9e1e9f62db2f6d2a7e388521f367ea1d2de47635ad201"
            )
            
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=512,
                temperature=self.temperature,
                top_p=0.7,
                top_k=50,
                repetition_penalty=1.2,
                stop=["<|eot_id|>", "<|eom_id|>"],
                stream=True
            )
            
            # Process streaming response
            generated_text = ""
            for token in response:
                if hasattr(token, 'choices'):
                    chunk = token.choices[0].delta.content
                    if chunk:
                        generated_text += chunk
            
            if not generated_text.strip():
                raise ValueError("No response generated")
                
            st.write(f"        Generated response length: {len(generated_text.split())} words")
            
            return generated_text.strip()
            
        except Exception as e:
            st.error(f"❌ Error in response generation: {str(e)}")
            raise

    def _calculate_quality_metrics(self, text: str) -> QualityMetrics:
        """Calculate quality metrics for generated text"""
        words = text.split()
        sentences = text.split('.')
        technical_terms = len(re.findall(r'\b[A-Z][a-z]*(?:\s+[A-Z][a-z]*)*\b', text))
        
        return QualityMetrics(
            complexity_score=sum(len(word) for word in words) / len(words),
            diversity_score=len(set(words)) / len(words),
            technical_density=technical_terms / len(words),
            avg_sentence_length=len(words) / len(sentences),
            word_count=len(words)
        )

    def _generate_unique_prompt(self, concept: str, topic: str) -> str:
        """Generate unique, contextual prompts"""
        # Base templates for different types of questions
        templates = [
            # Learning and understanding
            f"How can I get started with {concept} in {topic}?",
            f"What are the fundamental concepts of {concept}?",
            f"Could you explain {concept} in simple terms?",
            
            # Implementation and practice
            f"What's the best way to implement {concept}?",
            f"Can you show me a practical example of {concept}?",
            f"What are common patterns when using {concept}?",
            
            # Problem solving
            f"How do I debug issues with {concept}?",
            f"What are common mistakes to avoid with {concept}?",
            f"How can I optimize {concept} in my project?",
            
            # Best practices
            f"What are the best practices for {concept}?",
            f"How do experienced developers use {concept}?",
            f"What are some tips for working with {concept}?",
            
            # Integration
            f"How does {concept} integrate with other tools?",
            f"What's the workflow for using {concept}?",
            f"How can I combine {concept} with other features?"
        ]
        
        # Add variety with random prefixes and suffixes
        prefixes = [
            "",
            "Hey! ",
            "Quick question: ",
            "I need help: ",
            "Could you explain ",
            "I'm curious about ",
        ]
        
        suffixes = [
            "",
            " Any suggestions?",
            " I'd appreciate your help.",
            " Thanks in advance!",
            " I'm trying to learn more.",
            " I'm building a project."
        ]
        
        # Generate unique prompt
        template = random.choice(templates)
        prefix = random.choice(prefixes)
        suffix = random.choice(suffixes)
        
        return f"{prefix}{template}{suffix}".strip()

    def show_preview(self, prompt: str, response: str):
        with st.expander("Preview Generated Content", expanded=True):
            st.markdown("### Sample Preview")
            st.info(f"**Prompt:** {prompt}")
            st.success(f"**Response:** {response}")
            # Add quality metrics visualization
            st.markdown("### Quality Metrics")
            col1, col2, col3 = st.columns(3)
            metrics = self._calculate_quality_metrics(response)
            col1.metric("Complexity", f"{metrics.complexity_score:.2f}")
            col2.metric("Diversity", f"{metrics.diversity_score:.2f}")
            col3.metric("Technical Density", f"{metrics.technical_density:.2f}")

    def show_analytics(self, dataset: List[Dict]):
        st.markdown("## Generation Analytics")
        
        # Dataset statistics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Samples", len(dataset))
        col2.metric("Avg Response Length", 
                    f"{sum(len(d['response'].split()) for d in dataset) / len(dataset):.0f}")
        col3.metric("Unique Concepts", 
                    len(set(d['metadata']['concept'] for d in dataset)))
        
        # Quality distribution
        quality_scores = [d['metadata']['quality_metrics']['complexity_score'] 
                         for d in dataset]
        st.line_chart(quality_scores)

    def create_dashboard(self, dataset: List[Dict], current_step: int, total_steps: int):
        """Create real-time dashboard with charts and animations"""
        dashboard = st.container()
        
        with dashboard:
            st.markdown("## 📊 Generation Dashboard")
            
            # Progress metrics row
            col1, col2, col3, col4 = st.columns(4)
            
            # Progress
            progress_percentage = (current_step/total_steps)*100
            col1.metric(
                "Progress",
                f"{progress_percentage:.1f}%",
                f"{current_step}/{total_steps} rows"
            )
            st.progress(progress_percentage/100)
            
            if dataset:
                # Quality metrics over time - Fixed metric names
                history_df = pd.DataFrame([
                    {
                        'index': i,
                        'complexity_score': d['metadata']['quality_metrics']['complexity_score'],
                        'diversity_score': d['metadata']['quality_metrics']['diversity_score'],
                        'technical_density': d['metadata']['quality_metrics']['technical_density'],
                        'word_count': d['metadata']['quality_metrics']['word_count']
                    }
                    for i, d in enumerate(dataset)
                ])
                
                # Charts
                st.markdown("### 📈 Quality Metrics Trends")
                chart_col1, chart_col2 = st.columns(2)
                
                with chart_col1:
                    st.line_chart(
                        history_df[['complexity_score', 'diversity_score', 'technical_density']],
                        use_container_width=True
                    )
                    st.caption("Quality Metrics Over Time")
                
                with chart_col2:
                    # Word count distribution
                    fig = plt.figure(figsize=(10, 4))
                    plt.hist(history_df['word_count'], bins=20, color='skyblue', alpha=0.7)
                    plt.title("Response Length Distribution")
                    plt.xlabel("Word Count")
                    plt.ylabel("Frequency")
                    st.pyplot(fig)
                    plt.close()
                
                # Recent generations
                st.markdown("### 🔄 Recent Generations")
                recent_df = pd.DataFrame([
                    {
                        'Prompt': d['prompt'][:50] + "...",
                        'Response': d['response'][:100] + "...",
                        'Complexity': d['metadata']['quality_metrics']['complexity_score'],
                        'Diversity': d['metadata']['quality_metrics']['diversity_score'],
                        'Technical': d['metadata']['quality_metrics']['technical_density']
                    }
                    for d in dataset[-5:]  # Show last 5 entries
                ])
                
                st.dataframe(recent_df, use_container_width=True)
                
                # Quality targets - Fixed metric names
                st.markdown("### 🎯 Quality Targets")
                target_cols = st.columns(4)
                metrics = ['complexity_score', 'diversity_score', 'technical_density', 'word_count']
                display_names = ['Complexity', 'Diversity', 'Technical Density', 'Word Count']
                
                for i, (metric, display_name) in enumerate(zip(metrics, display_names)):
                    with target_cols[i]:
                        current_val = np.mean([
                            d['metadata']['quality_metrics'][metric]
                            for d in dataset
                        ])
                        target_val = self.quality_targets.targets[metric.replace('_score', '')]['target']
                        st.metric(
                            display_name,
                            f"{current_val:.2f}",
                            f"Target: {target_val}"
                        )

def export_dataset(dataset: List[Dict], export_format: str, include_fields: List[str]):
    """Convert dataset to the selected format and return bytes for download"""
    if export_format == "JSON":
        filtered_data = [{field: entry[field] for field in include_fields if field in entry} for entry in dataset]
        return json.dumps(filtered_data, indent=2).encode('utf-8')
    
    elif export_format == "CSV":
        df = pd.DataFrame(dataset)
        filtered_df = df[include_fields]
        return filtered_df.to_csv(index=False).encode('utf-8')
    
    elif export_format == "JSONL":
        filtered_data = [{field: entry[field] for field in include_fields if field in entry} for entry in dataset]
        return '\n'.join(json.dumps(entry) for entry in filtered_data).encode('utf-8')
    
    elif export_format == "Excel":
        df = pd.DataFrame(dataset)
        filtered_df = df[include_fields]
        output = BytesIO()
        filtered_df.to_excel(output, index=False)
        return output.getvalue()

class DatasetConfig:
    def __init__(self):
        self.formats = {
            "QA_PAIRS": "Question-Answer format for training",
            "INSTRUCTION": "Instruction-following format",
            "CHAT": "Multi-turn conversation format",
            "SYSTEM_HUMAN": "System and human interaction format",
            "RAG": "Retrieval-augmented generation format"
        }
        
        self.templates = {
            "QA_PAIRS": {
                "question": "Question: {query}",
                "answer": "Answer: {response}"
            },
            "INSTRUCTION": {
                "instruction": "Instruction: {query}",
                "response": "Response: {response}"
            },
            # ... other format templates
        }

def generate_enterprise_dataset(
    self,
    topic: str,
    concepts: List[str],
    num_samples: int,
    format_type: str = "QA_PAIRS",
    include_metadata: bool = True,
    quality_threshold: float = 0.8,
    diversity_check: bool = True
) -> List[Dict]:
    """Generate enterprise-quality dataset with specified format and quality controls"""
    
    dataset = []
    config = DatasetConfig()
    
    for concept in concepts:
        concept_data = []
        
        # Generate diverse examples for each concept
        while len(concept_data) < num_samples:
            # Generate base response
            entry = self._generate_base_entry(topic, concept)
            
            # Format according to template
            formatted_entry = self._format_entry(
                entry, 
                config.templates[format_type]
            )
            
            # Quality checks
            if self._meets_quality_standards(
                formatted_entry, 
                threshold=quality_threshold
            ):
                # Diversity check
                if not diversity_check or not self._is_duplicate(
                    formatted_entry, 
                    concept_data
                ):
                    concept_data.append(formatted_entry)
        
        dataset.extend(concept_data)
    
    return dataset

def main():
    st.title("🌟 Quasar Dataset Generator")

    # Sidebar configuration
    with st.sidebar:
        st.header("Model Configuration")
        
        # Model selection
        model_name = st.text_input(
            "Model Name",
            value="google/gemma-2-27b-it",
            help="Enter the name of the HuggingFace model to use"
        )

        # Dataset size configuration
        st.subheader("Dataset Size")
        target_rows = st.number_input(
            "Number of Rows",
            min_value=1,
            max_value=1000,
            value=100,
            help="Total number of dataset entries to generate"
        )

        # Topic categories
        st.subheader("Topic Categories")
        topic_categories = st.multiselect(
            "Select Categories",
            ["Technology", "Science", "Business", "Healthcare", "Education", 
             "Engineering", "Mathematics", "Programming", "Custom"],
            default=["Technology"],
            help="Choose relevant topic categories"
        )

        with st.expander("Quality Controls"):
            require_examples = st.checkbox(
                "Require Examples",
                value=True
            )
            content_filters = st.multiselect(
                "Content Filters",
                ["Technical Terms", "Citations", "Code Examples", "Mathematical Formulas"],
                default=["Technical Terms"]
            )
            
            diversity_score = st.slider(
                "Response Diversity",
                0.0, 1.0, 0.7,
                help="Higher values ensure more diverse responses"
            )

        # Export format and quality controls
        with st.expander("Export Settings"):
            export_format = st.selectbox(
                "Export Format",
                ["JSON", "CSV", "JSONL", "Excel"],
                help="Choose the output file format"
            )
            
            include_fields = st.multiselect(
                "Include Fields",
                ["prompt", "response", "context", "metadata", "system"],
                default=["prompt", "response", "metadata"],
                help="Select fields to include in export"
            )

    # Main area configuration
    # File upload section moved to main area
    st.header("Source Data")
    source_files = st.file_uploader(
        "Upload Source Files",
        type=['txt', 'csv', 'json', 'jsonl'],
        accept_multiple_files=True,
        help="Upload files containing knowledge you want to use"
    )

    st.header("Topic Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        topic = st.text_input(
            "Main Topic",
            value="",
            help="Enter the main topic (e.g., Quantum Physics, Machine Learning, etc.)"
        )
        
        st.subheader("Key Concept")
        concept = st.text_input("Enter your concept", key="concept")
        concepts = [concept] if concept else []
    
    with col2:
        subtopic = st.text_input(
            "Subtopic",
            value="",
            help="Enter a specific subtopic within your main topic"
        )
        
        context_info = st.text_area(
            "Additional Context",
            help="Provide any additional context or specific areas to focus on"
        )

    # Schema selection
    schema_template = st.selectbox(
        "Dataset Schema",
        ["Q&A Pairs", "Tutorial Format", "Problem-Solution", "Case Study", "Custom"],
        help="Choose the structure of your dataset entries"
    )

    # Generate button
    generate_button = st.button("Generate Dataset", type="primary")

    if generate_button and topic and concepts:
        try:
            st.write("🚀 Starting dataset generation process...")
            with st.spinner("Initializing generator..."):
                st.write(f"📌 Selected model: {model_name}")
                st.write(f"📌 Target rows: {target_rows}")
                st.write(f"📌 Topic: {topic}")
                st.write(f"📌 Concepts: {concepts}")
                
                # Initialize generator
                generator = EnhancedDatasetGenerator(
                    model_name=model_name,
                )
                
                # Add uploaded files as context if available
                if source_files:
                    with st.spinner("Processing knowledge files..."):
                        num_examples = generator.add_context_data(source_files)
                        st.success(f"✅ Added {num_examples} examples as context knowledge")
                
                # Generate dataset
                dataset = generator.generate_dataset(
                    topic=topic,
                    concepts=concepts,
                    num_samples=target_rows
                )
                
                st.success("Dataset generated successfully!")
                
                # Display results
                for entry in dataset:
                    st.write("---")
                    st.write(f"Topic: {entry['metadata']['topic']}")
                    st.write(f"Concept: {entry['metadata']['concept']}")
                    st.write(f"Response: {entry['response']}")

                # Add download button
                st.write("---")
                st.write("### Download Dataset")
                
                # Prepare download data
                file_extension = {
                    "JSON": "json",
                    "CSV": "csv",
                    "JSONL": "jsonl",
                    "Excel": "xlsx"
                }[export_format]
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"dataset_{topic.lower().replace(' ', '_')}_{timestamp}.{file_extension}"
                
                download_data = export_dataset(dataset, export_format, include_fields)
                
                # Create download button
                st.download_button(
                    label=f"Download {export_format} Dataset",
                    data=download_data,
                    file_name=filename,
                    mime={
                        "JSON": "application/json",
                        "CSV": "text/csv",
                        "JSONL": "application/jsonl",
                        "Excel": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    }[export_format]
                )

        except Exception as e:
            st.error(f"Generation failed: {str(e)}")
    elif generate_button:
        st.warning("Please fill in all required f   ields (Topic and at least one Concept)")

if __name__ == "__main__":
    main()
