from flask import Flask, render_template, request
from model import llm_model
import markdown
import textwrap

app = Flask(__name__)

def md_to_html(md_text: str) -> str:
    """Change markdown text to HTML. The text is dedented before conversion.

    Args:
        md_text (str): Text in markdown format.

    Returns:
        _type_: Text in HTML format.
    """

    md_text = textwrap.dedent(md_text)
    return markdown.markdown(md_text, extensions=['tables'])    


@app.route('/', methods=['GET', 'POST'])
def home():

    if request.method == 'POST':
        user_input = request.form.get('user-input')
        google_api_key = 'your-api-key'

        llm_response = llm_model(user_input, google_api_key, top_k=70, temperature=0.8)
        answer = md_to_html(llm_response["answer"])
        context = md_to_html(llm_response["context"])

        # Some formatting for better readability
        context = context.replace('Subtopic:', '<br><br><b>Subtopic:</b><br><br>')
        context = context.replace('Page extracted:', '<b>Page extracted:</b>')
        answer = answer.replace('Source:', '<br>Source:')


        return render_template('main.html',
                               context=context,
                               answer=answer,
                               user_input=user_input)
    
    return render_template('main.html')

if __name__ == '__main__':
    app.run(debug=True)
