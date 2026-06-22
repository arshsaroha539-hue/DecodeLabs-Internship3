"""
Tech Stack Recommender System
DecodeLabs - AI Industrial Training | Project 3
Uses TF-IDF + Cosine Similarity (Content-Based Filtering)
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_dataset(filepath="raw_skills.csv"):
    """Step 1 - INGESTION (Item Side): Load job roles dataset."""
    df = pd.read_csv(filepath)
    df["skills"] = df["skills"].str.lower().str.replace("_", " ")
    return df


def build_tfidf_matrix(df):
    """Build TF-IDF matrix from item dataset."""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df["skills"])
    return vectorizer, tfidf_matrix


def build_user_profile(user_skills, vectorizer):
    """Step 1 - INGESTION (User Side): Convert user skills to TF-IDF vector."""
    user_input = " ".join([skill.lower() for skill in user_skills])
    user_vector = vectorizer.transform([user_input])
    return user_vector


def score_and_rank(user_vector, tfidf_matrix, df, top_n=3):
    """
    Step 2 - SCORING: Compute cosine similarity scores.
    Step 3 - SORTING: Sort by descending score.
    Step 4 - FILTERING: Return Top-N results.
    """
    # Step 2: Score
    scores = cosine_similarity(user_vector, tfidf_matrix).flatten()

    # Step 3: Sort
    sorted_indices = np.argsort(scores)[::-1]

    # Step 4: Filter - Top N
    results = []
    for idx in sorted_indices[:top_n]:
        results.append({
            "rank": len(results) + 1,
            "job_role": df.iloc[idx]["job_role"],
            "match_score": round(float(scores[idx]) * 100, 2),
            "skills": df.iloc[idx]["skills"]
        })
    return results


def get_user_skills():
    """Collect at least 3 skills from the user."""
    print("\n" + "="*60)
    print("   TECH STACK RECOMMENDER — DecodeLabs AI Project 3")
    print("="*60)
    print("\nEnter your skills one by one (minimum 3 required).")
    print("Press ENTER after each skill. Type 'done' when finished.\n")

    skills = []
    while True:
        skill = input(f"  Skill {len(skills)+1}: ").strip()
        if skill.lower() == "done":
            if len(skills) < 3:
                print("  ⚠️  Please enter at least 3 skills.\n")
            else:
                break
        elif skill:
            skills.append(skill)
            if len(skills) >= 3:
                print("  (You can add more or type 'done' to get recommendations)")
    return skills


def display_results(results, user_skills):
    """Display the top recommendations."""
    print("\n" + "="*60)
    print(f"  YOUR SKILLS: {', '.join(user_skills)}")
    print("="*60)
    print(f"\n  🎯 TOP {len(results)} RECOMMENDED CAREER PATHS:\n")

    for r in results:
        bar_length = int(r["match_score"] / 5)
        bar = "█" * bar_length + "░" * (20 - bar_length)
        print(f"  #{r['rank']}  {r['job_role']}")
        print(f"      Match: [{bar}] {r['match_score']}%")
        print(f"      Key Skills: {r['skills'][:60]}...")
        print()

    print("="*60)
    print("  💡 Tip: Add more specific skills for better accuracy!")
    print("="*60 + "\n")


def main():
    # Load data and build TF-IDF model
    df = load_dataset("/home/claude/raw_skills.csv")
    vectorizer, tfidf_matrix = build_tfidf_matrix(df)

    while True:
        # Get user input (min 3 skills)
        user_skills = get_user_skills()

        # Build user vector
        user_vector = build_user_profile(user_skills, vectorizer)

        # Check for cold start problem
        if user_vector.nnz == 0:
            print("\n  ⚠️  No matching skills found in our database.")
            print("  Try skills like: Python, SQL, AWS, JavaScript, Docker...\n")
        else:
            # Score, Sort, Filter
            results = score_and_rank(user_vector, tfidf_matrix, df, top_n=3)
            display_results(results, user_skills)

        again = input("  Try again with different skills? (yes/no): ").strip().lower()
        if again != "yes":
            print("\n  Thank you for using Tech Stack Recommender! 🚀\n")
            break


if __name__ == "__main__":
    main()
