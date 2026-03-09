import os
from openai import OpenAI


def explain_issue(issue_data):
    """
    Generate AI explanation for detected ML issues.
    """

    try:

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        prompt = f"""
        A machine learning diagnostic system detected the following issue:

        {issue_data}

        Explain this issue in simple terms and suggest possible fixes.
        """

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        explanation = response.choices[0].message.content

        return explanation

    except Exception as e:

        return f"AI explanation unavailable: {str(e)}"