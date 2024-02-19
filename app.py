from flask import (
    Flask,
    render_template,
    redirect,
    url_for,
    flash,
    request,
    jsonify,
    Response,
)
from flask import (
    Flask,
    render_template,
    redirect,
    url_for,
    flash,
    request,
    jsonify,
    Response,
    send_from_directory,
    abort,
)
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_login import (
    LoginManager,
    UserMixin,
    login_user,
    login_required,
    logout_user,
    current_user,
)
from multiprocessing import Process
import numpy as np

from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime, timedelta
from flask_wtf import FlaskForm
from wtforms.validators import DataRequired, EqualTo
from wtforms.fields import (
    StringField,
    SubmitField,
    BooleanField,
    DateField,
    TextAreaField,
    ColorField,
    FileField,
)
from wtforms_sqlalchemy.fields import QuerySelectField, QuerySelectMultipleField

import hashlib

from werkzeug.utils import secure_filename
from sqlalchemy.dialects import sqlite

import requests
import os

from wtforms import SelectField
import json
import re

from flask import Flask, request, Response, render_template
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.dialects import sqlite
from werkzeug.security import check_password_hash, generate_password_hash
from flask_migrate import Migrate

from sqlalchemy import func

import os


from flask_cors import CORS
from flask_socketio import SocketIO, emit
from flask import redirect, url_for
from functools import wraps

from flask_login import current_user
from functools import wraps
from flask_socketio import disconnect
import shutil

from dotenv import load_dotenv

from uuid import uuid4

from sqlalchemy import event

load_dotenv()

app = Flask(__name__)
socketio = SocketIO(app)

CORS(app, resources={r"/e": {"origins": "*"}})


database_path = os.path.join(os.getcwd(), 'data', 'database.sqlite')
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{database_path}"
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY")
app.config["UPLOAD_FOLDER"] = "./data/uploads"

db = SQLAlchemy(app)
migrate = Migrate(app, db)

login_manager = LoginManager()
login_manager.init_app(app)

per_page = 10

themes = {
    "dark": {
        "background_color": "#1f1f1f",
        "pane_color": "#171717",
        "border_color": "#505050",
        "text_color": "#CCCCCC",
        "text_color_secondary": "#999999",
        "button_text_color": "#CCCCCC",
        "button_background": "#161616",
        "button_hover_background": "#1b1b1b",
    },
    "monokai": {
        "background_color": "#272822",
        "pane_color": "#383830",
        "border_color": "#75715E",
        "text_color": "#F8F8F2",
        "text_color_secondary": "#5E5E56",
        "button_text_color": "#F8F8F2",
        "button_text_color": "#F8F8F2",
        "button_background": "#49483E",
        "button_hover_background": "#75715E",
    },
    "solarized_light": {
        "background_color": "#FDF6E3",
        "pane_color": "#EEE8D5",
        "border_color": "#93A1A1",
        "text_color": "#586E75",
        "text_color_secondary": "#C1C3B8",
        "button_text_color": "#586E75",
        "button_text_color": "#FDF6E3",
        "button_background": "#657B83",
        "button_hover_background": "#839496",
    },
    "dracula": {
        "background_color": "#282a36",
        "pane_color": "#44475a",
        "border_color": "#6272a4",
        "text_color": "#f8f8f2",
        "text_color_secondary": "#999999",
        "button_background": "#44475a",
        "button_hover_background": "#6272a4",
        "button_text_color": "#f1fa8c",
    },
    "nord": {
        "background_color": "#2E3440",
        "pane_color": "#3B4252",
        "border_color": "#4C566A",
        "text_color": "#D8DEE9",
        "text_color_secondary": "#E5E9F0",
        "button_background": "#3B4252",
        "button_hover_background": "#4C566A",
        "button_text_color": "#ECEFF4",
    },
    "matrix": {
        "background_color": "#0D0208",
        "pane_color": "#003B00",
        "border_color": "#004F00",
        "text_color": "#00FF41",
        "text_color_secondary": "#00FF41",
        "button_background": "#003B00",
        "button_hover_background": "#004F00",
        "button_text_color": "#00FF41",
    },
    "light": {
        "background_color": "#F5F5F5",
        "pane_color": "#E8E8E8",
        "border_color": "#CCCCCC",
        "text_color": "#333333",
        "text_color_secondary": "#999999",
        "button_background": "#E8E8E8",
        "button_hover_background": "#D6D6D6",
        "button_text_color": "#333333",
    },
}


@app.after_request
def set_cache_header(response):
    if request.path.startswith('/static'):
        response.headers['Cache-Control'] = 'public, max-age=31536000'  # 1 year
    return response

@app.context_processor
def inject_theme():
    selected_theme = themes[
        (
            (current_user.theme if current_user.theme else "light")
            if current_user.is_authenticated
            else "light"
        )
    ]
    return dict(theme=selected_theme)


def human_readable_date(value):
    now = datetime.now()
    diff = now - value

    days = diff.days
    if days < 1:
        hours = diff.seconds // 3600
        if hours < 1:
            minutes = diff.seconds // 60
            if minutes < 1:
                return "just now"
            else:
                return f"{minutes} minutes ago"
        else:
            return f"{hours} hours ago"
    elif days < 30:
        return f"{days} days ago"
    else:
        return value.strftime("%d-%m-%Y")  # Date in dd-mm-yyyy format


def human_readable_size(num_bytes):
    for unit in ["bytes", "KB", "MB", "GB", "TB", "PB", "EB", "ZB"]:
        if num_bytes < 1024.0:
            return f"{num_bytes:3.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} YB"


def send_telegram_to_user(user, message):
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = user.telegram_id
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = {"chat_id": chat_id, "text": message}
    response = requests.post(url, data=data)
    return response.json()


@app.template_filter("currency")
def currency_filter(amount):
    return f"€{amount:,.2f}"


@app.template_filter("date")
def date_filter(date):
    return date.strftime("%d-%m-%Y")


@app.template_filter("phone")
def phone_filter(phone):
    # replace 00 at start with +
    if phone.startswith("00"):
        phone = "+" + phone[2:]
        phone = phone[:3] + " " + phone[3:]
        phone = phone[:7] + " " + phone[7:]
        phone = phone[:11] + " " + phone[11:]
    return phone


def is_contact(instance):
    return isinstance(instance, Contact)


def is_user(instance):
    return isinstance(instance, User)


def is_date(instance):
    return isinstance(instance, datetime)


def is_file(instance):
    return isinstance(instance, File)


def are_tags(list_):
    if isinstance(list_, list):
        return all(isinstance(item, Tag) for item in list_)


def is_company(instance):
    return isinstance(instance, Company)


app.jinja_env.globals["is_contact"] = is_contact
app.jinja_env.globals["is_user"] = is_user
app.jinja_env.globals["is_date"] = is_date
app.jinja_env.globals["is_file"] = is_file
app.jinja_env.globals["is_company"] = is_company
app.jinja_env.filters["human_readable_date"] = human_readable_date
app.jinja_env.filters["human_readable_size"] = human_readable_size
app.jinja_env.globals["are_tags"] = are_tags


@app.template_filter("markdown")
def markdown_to_html(markdown_text):
    html_lines = []
    in_list = False

    for line in markdown_text.split("\n"):
        # Headings
        if line.startswith("# "):
            html_lines.append(f"<h1>{line[2:]}</h1>")

        # Second headings
        elif line.startswith("## "):
            html_lines.append(f"<h2>{line[3:]}</h2>")

        # Third headings
        elif line.startswith("### "):
            html_lines.append(f"<h3>{line[4:]}</h3>")

        # Unordered lists
        elif line.startswith("- "):
            if not in_list:
                html_lines.append("<ul>")
                in_list = True
            html_lines.append(f"<li>{line[2:]}</li>")

        # Links and Images
        else:
            line = convert_links_and_images(line)
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            if line.strip():
                html_lines.append(line)
            else:
                html_lines.append("<br>")

    if in_list:
        html_lines.append("</ul>")

    return "\n".join(html_lines)


def strip_html_tags(input_string):
    # Regular expression for finding HTML tags
    cleanr = re.compile("<.*?>")
    # Replacing HTML tags with an empty string
    cleantext = re.sub(cleanr, "", input_string)
    return cleantext


def convert_links_and_images(line):
    # Convert images ![alt text](URL)
    line = re.sub(r"!\[([^\]]+)\]\(([^\)]+)\)", r'<img alt="\1" src="\2">', line)
    # Convert links [link text](URL)
    line = re.sub(r"\[([^\]]+)\]\(([^\)]+)\)", r'<a href="\2">\1</a>', line)
    return line


def query_openai(messages, json_mode=False):
    api_key = os.getenv("OPENAI_API_KEY")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    data = {"model": "gpt-4", "messages": messages}

    if json_mode:
        data["response_format"] = {"type": "json_object"}

    response = requests.post(
        "https://api.openai.com/v1/chat/completions", json=data, headers=headers
    )
    return response.json()





def get_openai_embedding(text):
    api_url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
    }
    data = {"input": text, "model": "text-embedding-3-small"}

    response = requests.post(api_url, json=data, headers=headers)
    embedding = response.json()["data"][0]["embedding"]
    return embedding


def process_file(file_id):
    from pdfminer.high_level import extract_text

    file_ = File.query.get(file_id)

    # extract text from pdf
    text = extract_text(os.path.join(app.config["UPLOAD_FOLDER"], file_.filename))

    # chunk text
    def create_chunks(text, chunk_size=512, overlap=20):
        lines = text.split("\n")
        chunks = []
        current_chunk = []
        word_count = 0

        for line in lines:
            words = line.split()
            for word in words:
                current_chunk.append(word)
                word_count += 1

                if word_count >= chunk_size:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = current_chunk[-overlap:]
                    word_count = len(current_chunk)

            current_chunk.append("\n")  # Preserve the newline at the end of each line

        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(" ".join(current_chunk).strip())

        return chunks

    chunks = create_chunks(text)

    print("Chunking and embedding...")
    for chunk in chunks:
        c = Chunk()
        c.text = chunk
        c.file = file_
        db.session.add(c)

        embedding = get_openai_embedding(chunk)
        e = Embedding()
        e.vector = embedding
        e.chunk = c
        db.session.add(e)

    file_.status = "processed"
    db.session.commit()


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class BaseModel(db.Model):
    __abstract__ = True

    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow)




class HasImage:
    image = db.Column(db.String)

    @property
    def image_url(self):
        if self.image:
            return url_for("uploaded_file", filename=self.image)
        else:
            return url_for("static", filename="images/user.jpeg")

    def upload_image(self, image):
        # image comes from request.files['image']
        if image:
            filename = f"{uuid4()}.png"
            image.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            self.image = filename
            db.session.commit()


class User(UserMixin, BaseModel, HasImage):
    name = db.Column(db.String)
    email = db.Column(db.String, unique=True)
    password = db.Column(db.String)
    telegram_id = db.Column(db.String)
    theme = db.Column(db.String, default="light")
    is_assistant = db.Column(db.Boolean, default=False)
    assistant_system_prompt = db.Column(db.String)
    assistant_actions = db.Column(db.String)
    assistant_description = db.Column(db.String)
    assistant_example_prompts = db.Column(db.String)

    def __repr__(self):
        return f"<User {self.name}>"

    def __str__(self):
        return self.name

    def set_password(self, password):
        self.password = generate_password_hash(password)

    def check_password(self, password):
        if self.is_assistant:
            return False
        return check_password_hash(self.password, password)


class Folder(BaseModel):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String)
    parent_id = db.Column(db.Integer, db.ForeignKey("folder.id"))
    parent = db.relationship("Folder", backref="children", remote_side=[id])

    def __repr__(self):
        return f"<Folder {self.name}>"

    def __str__(self):
        return self.name


class FileSecret(BaseModel):
    file_id = db.Column(db.Integer, db.ForeignKey("file.id"))
    secret = db.Column(db.String)
    interaction_id = db.Column(db.Integer, db.ForeignKey("interaction.id"))
    interaction = db.relationship("Interaction", backref="secrets")
    contact_id = db.Column(db.Integer, db.ForeignKey("contact.id"))
    contact = db.relationship("Contact", backref="secrets")
    expires_at = db.Column(db.DateTime)

    def update_secret(self):
        self.secret = hashlib.sha256(os.urandom(60)).hexdigest()


class File(BaseModel):
    original_filename = db.Column(db.String)
    filename = db.Column(db.String)
    file_type = db.Column(db.String)  # pdf, image, video, audio, docx, xlsx, pptx, txt
    md5 = db.Column(db.String)
    size = db.Column(db.Integer)
    mime_type = db.Column(db.String)
    uploaded_by_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    uploaded_by = db.relationship("User", backref="files")
    folder_id = db.Column(db.Integer, db.ForeignKey("folder.id"))
    folder = db.relationship("Folder", backref="files")
    text = db.Column(db.Text)
    chunks = db.relationship("Chunk", cascade="all, delete-orphan", backref="file")
    secrets = db.relationship("FileSecret", cascade="all, delete-orphan")
    status = db.Column(db.String)  # processing, processed, failed


class Chunk(BaseModel):
    start = db.Column(db.Integer)
    end = db.Column(db.Integer)
    start_page = db.Column(db.Integer)
    end_page = db.Column(db.Integer)
    text = db.Column(db.String)
    file_id = db.Column(db.Integer, db.ForeignKey("file.id"))
    embeddings = db.relationship(
        "Embedding", cascade="all, delete-orphan", backref="chunk"
    )


class Embedding(BaseModel):
    chunk_id = db.Column(db.Integer, db.ForeignKey("chunk.id"))
    vector = db.Column(sqlite.JSON)


class Contact(BaseModel, HasImage):
    name = db.Column(db.String)
    email = db.Column(db.String)
    phone = db.Column(db.String)
    title = db.Column(db.String)
    company_id = db.Column(db.Integer, db.ForeignKey("company.id"))
    company = db.relationship("Company", backref="contacts")
    notes = db.Column(db.String)
    notes_summary = db.Column(db.String)
    notes_summary_one_line = db.Column(db.String)

    def __repr__(self):
        return "<Contact %r>" % self.name

    def __str__(self):
        return self.name
    
    @property
    def last_interaction_at(self):
        if self.interactions:
            return self.interactions[-1].date
        else:
            return None


class Task(BaseModel):
    name = db.Column(db.String)
    due_date = db.Column(db.DateTime)
    status = db.Column(db.String, default="todo")  # todo, doing, done, backlog
    contact_id = db.Column(db.Integer, db.ForeignKey("contact.id"))
    contact = db.relationship("Contact", backref="tasks")
    assigned_to_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    assigned_to = db.relationship("User", backref="tasks")
    interaction_id = db.Column(db.Integer, db.ForeignKey("interaction.id"))
    interaction = db.relationship("Interaction", backref="tasks")

    def __repr__(self):
        return f"<Task {self.name}>"




def task_insert_listener(mapper, connection, target):
    # Instead of committing within the listener, add the notification object to the session.
    # Ensure the session is committed later in your application flow.
    if target.assigned_to:
        notification = Notification()
        notification.user = target.assigned_to
        notification.message = f"You have been assigned to a new task: {target.name}"
        # Add the notification to the session but do not commit here.
        db.session.add(notification)


# Attach the listener to the Task model for both insert and update events
event.listen(Task, "after_insert", task_insert_listener)


interaction_opportunity_association = db.Table(
    "interaction_opportunity",
    db.Column(
        "interaction_id", db.Integer, db.ForeignKey("interaction.id"), primary_key=True
    ),
    db.Column(
        "opportunity_id", db.Integer, db.ForeignKey("opportunity.id"), primary_key=True
    ),
)


class Company(BaseModel, HasImage):
    name = db.Column(db.String)
    location = db.Column(db.String)
    website = db.Column(db.String)

    def __str__(self):
        return self.name


class Interaction(BaseModel):
    contact_id = db.Column(db.Integer, db.ForeignKey("contact.id"))
    contact = db.relationship("Contact", backref="interactions")
    date = db.Column(db.DateTime)
    notes = db.Column(db.String)
    notes_summary = db.Column(db.String)
    notes_summary_one_line = db.Column(db.String)
    type_ = db.Column(db.String)  # call, email, other
    logged_by_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    logged_by = db.relationship("User", backref="interactions")
    opportunities = db.relationship(
        "Opportunity",
        secondary=interaction_opportunity_association,
        backref=db.backref("interactions", lazy="dynamic"),
    )

    def __repr__(self):
        return f"<Interaction {self.contact.name} {self.date}>"


class Opportunity(BaseModel):
    contact_id = db.Column(db.Integer, db.ForeignKey("contact.id"))
    contact = db.relationship("Contact", backref="opportunities")
    date = db.Column(db.DateTime)
    name = db.Column(db.String)
    notes = db.Column(db.String)
    value = db.Column(db.Integer)

    def __repr__(self):
        return f"{self.name}"


# set up many-to-many relationship between users and conversations
user_conversation_association = db.Table(
    "user_conversation",
    db.Column("user_id", db.Integer, db.ForeignKey("user.id"), primary_key=True),
    db.Column(
        "conversation_id",
        db.Integer,
        db.ForeignKey("conversation.id"),
        primary_key=True,
    ),
)


class Conversation(BaseModel):
    participants = db.relationship(
        "User",
        secondary=user_conversation_association,
        backref=db.backref("conversations", lazy="dynamic"),
    )

    def __repr__(self):
        return f"{self.messages}"


class Message(BaseModel):
    conversation_id = db.Column(db.Integer, db.ForeignKey("conversation.id"))
    conversation = db.relationship("Conversation", backref="messages")
    sender_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    sender = db.relationship("User", backref="messages")
    text = db.Column(db.String)

    def __repr__(self):
        return f"{self.text}"


class Notification(BaseModel):
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    user = db.relationship("User", backref="notifications")
    message = db.Column(db.String)
    read = db.Column(db.Boolean, default=False)
    interaction_id = db.Column(db.Integer, db.ForeignKey("interaction.id"))
    interaction = db.relationship("Interaction", backref="notifications")
    opportunity_id = db.Column(db.Integer, db.ForeignKey("opportunity.id"))
    opportunity = db.relationship("Opportunity", backref="notifications")
    task_id = db.Column(db.Integer, db.ForeignKey("task.id"))
    task = db.relationship("Task", backref="notifications")

    def __repr__(self):
        return f"{self.message}"


from flask_wtf.file import FileAllowed


class ContactForm(FlaskForm):
    name = StringField("Name", validators=[DataRequired()])
    email = StringField("Email")
    phone = StringField("Phone")
    notes = TextAreaField("Notes")
    tags = QuerySelectMultipleField(
        "Tags", query_factory=lambda: Tag.query.order_by(Tag.name).all()
    )
    company = QuerySelectField(
        "Company", query_factory=lambda: Company.query.order_by(Company.name).all()
    )
    image = FileField(
        "Image", validators=[FileAllowed(["png"], "Only .png files allowed")]
    )
    submit = SubmitField("Save")


class CompanyForm(FlaskForm):
    name = StringField("Name", validators=[DataRequired()])
    location = StringField("Location")
    website = StringField("Website")
    image = FileField(
        "Image", validators=[FileAllowed(["png"], "Only .png files allowed")]
    )
    submit = SubmitField("Save")


class InteractionForm(FlaskForm):
    contact = QuerySelectField(
        "Contact", query_factory=lambda: Contact.query.order_by(Contact.name).all()
    )
    opportunities = QuerySelectMultipleField(
        "Opportunities", query_factory=lambda: Opportunity.query.all()
    )
    date = DateField("Date", default=datetime.utcnow)
    notes = TextAreaField("Notes")
    type_ = SelectField(
        "Type", choices=[("call", "Call"), ("email", "Email"), ("other", "Other")]
    )
    submit = SubmitField("Save")


class OpportunityForm(FlaskForm):
    contact = QuerySelectField(
        "Contact", query_factory=lambda: Contact.query.order_by(Contact.name).all()
    )
    date = DateField("Date", default=datetime.utcnow)
    name = StringField("Name")
    notes = TextAreaField("Notes")
    value = StringField("Value")
    submit = SubmitField("Save")


class LoginForm(FlaskForm):
    email = StringField("Email", validators=[DataRequired()])
    password = StringField("Password", validators=[DataRequired()])
    submit = SubmitField("Login")


class TaskForm(FlaskForm):
    name = StringField("Name", validators=[DataRequired()])
    due_date = DateField("Due Date", default=datetime.utcnow)
    status = SelectField(
        "Status",
        choices=[
            ("todo", "Todo"),
            ("doing", "Doing"),
            ("done", "Done"),
            ("backlog", "Backlog"),
        ],
    )
    contact = QuerySelectField(
        "Contact",
        query_factory=lambda: Contact.query.all(),
        allow_blank=True,
        blank_text="None",
    )
    assigned_to = QuerySelectField(
        "Assigned To",
        query_factory=lambda: User.query.all(),
        allow_blank=True,
        blank_text="None",
    )
    interaction = QuerySelectField(
        "Interaction",
        query_factory=lambda: Interaction.query.all(),
        allow_blank=True,
        blank_text="None",
    )
    tags = QuerySelectMultipleField("Tags", query_factory=lambda: Tag.query.all())
    submit = SubmitField("Save")


class UserForm(FlaskForm):
    name = StringField("Name", validators=[DataRequired()])
    email = StringField("Email", validators=[DataRequired()])
    password = StringField("Password", validators=[DataRequired()])
    password2 = StringField(
        "Password", validators=[DataRequired(), EqualTo("password")]
    )
    telegram_id = StringField("Telegram ID")
    theme = SelectField(
        "Theme",
        choices=[
            ("dark", "Dark"),
            ("light", "Light"),
            ("monokai", "Monokai"),
            ("solarized_light", "Solarized Light"),
            ("dracula", "Dracula"),
            ("nord", "Nord"),
            ("matrix", "Matrix"),
            ("barbie", "Barbie"),
        ],
    )
    is_assistant = BooleanField("Is Assistant")
    assistant_system_prompt = TextAreaField("Assistant System Prompt")
    assistant_actions = TextAreaField("Assistant Actions")
    assistant_description = TextAreaField("Assistant Description")
    assistant_example_prompts = TextAreaField("Assistant Example Prompts")
    image = FileField(
        "Image", validators=[FileAllowed(["png"], "Only .png files allowed")]
    )
    submit = SubmitField("Save")


campaign_contact_association = db.Table(
    "campaign_contact_association",
    db.Column(
        "campaign_id", db.Integer, db.ForeignKey("campaign.id"), primary_key=True
    ),
    db.Column("contact_id", db.Integer, db.ForeignKey("contact.id"), primary_key=True),
)


class Comment(BaseModel):
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    user = db.relationship("User", backref="comments")
    text = db.Column(db.String)
    contact_id = db.Column(db.Integer, db.ForeignKey("contact.id"))
    contact = db.relationship("Contact", backref="comments")
    opportunity_id = db.Column(db.Integer, db.ForeignKey("opportunity.id"))
    opportunity = db.relationship("Opportunity", backref="comments")
    interaction_id = db.Column(db.Integer, db.ForeignKey("interaction.id"))
    interaction = db.relationship("Interaction", backref="comments")
    task_id = db.Column(db.Integer, db.ForeignKey("task.id"))
    task = db.relationship("Task", backref="comments")

    def __repr__(self):
        return f"<Comment {self.content}>"


# many to many between tags and contacts, opportunities, interactions and tasks
tags_contacts = db.Table(
    "tag_contact_association",
    db.Column("tag_id", db.Integer, db.ForeignKey("tag.id"), primary_key=True),
    db.Column("contact_id", db.Integer, db.ForeignKey("contact.id"), primary_key=True),
)

tags_opportunities = db.Table(
    "tag_opportunity_association",
    db.Column("tag_id", db.Integer, db.ForeignKey("tag.id"), primary_key=True),
    db.Column(
        "opportunity_id", db.Integer, db.ForeignKey("opportunity.id"), primary_key=True
    ),
)

tags_interactions = db.Table(
    "tag_interaction_association",
    db.Column("tag_id", db.Integer, db.ForeignKey("tag.id"), primary_key=True),
    db.Column(
        "interaction_id", db.Integer, db.ForeignKey("interaction.id"), primary_key=True
    ),
)

tags_tasks = db.Table(
    "tag_task_association",
    db.Column("tag_id", db.Integer, db.ForeignKey("tag.id"), primary_key=True),
    db.Column("task_id", db.Integer, db.ForeignKey("task.id"), primary_key=True),
)

tags_companies = db.Table(
    "tag_company_association",
    db.Column("tag_id", db.Integer, db.ForeignKey("tag.id"), primary_key=True),
    db.Column("company_id", db.Integer, db.ForeignKey("company.id"), primary_key=True),
)


class Tag(BaseModel):
    name = db.Column(db.String)
    contacts = db.relationship(
        "Contact",
        secondary="tag_contact_association",
        backref=db.backref("tags", lazy="select"),
    )
    opportunities = db.relationship(
        "Opportunity",
        secondary="tag_opportunity_association",
        backref=db.backref("tags", lazy="select"),
    )
    interactions = db.relationship(
        "Interaction",
        secondary="tag_interaction_association",
        backref=db.backref("tags", lazy="select"),
    )
    tasks = db.relationship(
        "Task",
        secondary="tag_task_association",
        backref=db.backref("tags", lazy="select"),
    )
    companies = db.relationship(
        "Company",
        secondary="tag_company_association",
        backref=db.backref("tags", lazy="select"),
    )
    color = db.Column(db.String(6))

    def __repr__(self):
        return f"<Tag {self.name}>"

    @property
    def text_color(self):
        r = int(self.color[:2], 16)
        g = int(self.color[2:4], 16)
        b = int(self.color[4:], 16)
        brightness = (r * 299 + g * 587 + b * 114) / 1000
        if brightness > 155:
            return "black"
        else:
            return "white"

    def __str__(self):
        return self.name


class Redirect(BaseModel):
    id = db.Column(db.Integer, primary_key=True)
    url = db.Column(db.String)
    uid = db.Column(db.String)
    comment = db.Column(db.String)


class Event(BaseModel):
    id = db.Column(db.Integer, primary_key=True)
    data = db.Column(sqlite.JSON)
    session = db.Column(db.String)
    at = db.Column(db.DateTime, server_default=db.func.now())
    ip = db.Column(db.String)
    url = db.Column(db.String)


class TagForm(FlaskForm):
    name = StringField("Name", validators=[DataRequired()])
    color = ColorField("Color", validators=[DataRequired()])
    submit = SubmitField("Save")


class Campaign(BaseModel):
    name = db.Column(db.String)
    contacts = db.relationship(
        "Contact",
        secondary="campaign_contact_association",
        backref=db.backref("campaigns", lazy="dynamic"),
    )


class CampaignForm(FlaskForm):
    name = StringField("Name", validators=[DataRequired()])
    contacts = QuerySelectMultipleField(
        "Contacts", query_factory=lambda: Contact.query.all()
    )
    submit = SubmitField("Save")


class Calendar(BaseModel):
    name = db.Column(db.String)
    created_by_id = db.Column(db.Integer, db.ForeignKey("user.id"))
    created_by = db.relationship("User", backref="calendars")


class CalendarEvent(BaseModel):
    name = db.Column(db.String)
    date = db.Column(db.DateTime)
    calendar_id = db.Column(db.Integer, db.ForeignKey("calendar.id"))
    calendar = db.relationship("Calendar", backref="events")

    def __repr__(self):
        return f"<CalendarEvent {self.name}>"


class FolderForm(FlaskForm):
    name = StringField("Name", validators=[DataRequired()])
    submit = SubmitField("Save")


class ShareFileForm(FlaskForm):
    contact = QuerySelectField("Contact", query_factory=lambda: Contact.query.all())
    expires_at = DateField("Expires At", default=datetime.utcnow() + timedelta(days=30))
    submit = SubmitField("Save")


class RedirectForm(FlaskForm):
    url = StringField("URL", validators=[DataRequired()])
    comment = StringField("Comment")
    submit = SubmitField("Save")


login_manager.login_view = "login"


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    form = LoginForm()

    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data.lower()).first()

        if not user:
            flash("Invalid email or password")
            print("Invalid email or password")
            return render_template("login.html", form=form)

        if user and user.check_password(form.password.data):
            login_user(user)
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid email or password")
            print("Invalid email or password")
            return render_template("login.html", form=form)

    return render_template("login.html", form=form)


@app.route("/")
def dashboard():
    db.create_all()

    # Check if there are users, if not, create admin user
    if not User.query.first():
        user = User()
        user.name = "Admin"
        user.email = "admin@admin.com"
        user.set_password("admin")
        db.session.add(user)
        db.session.commit()
        db.session.flush()

    # redirect to companies list
    return redirect(url_for("view_list", object_name="companies"))


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


@app.route("/<object_name>/list/")
@login_required
def view_list(object_name):
    page_details = {
        "companies": {"title": "Companies", "icon": "companies"},
        "contacts": {"title": "Contacts", "icon": "contacts"},
        "interactions": {"title": "Interactions", "icon": "interactions"},
        "opportunities": {"title": "Opportunities", "icon": "opportunities"},
        "tasks": {"title": "Tasks", "icon": "tasks"},
        "tags": {"title": "Tags", "icon": "tags"},
        "calendar": {"title": "Calendar", "icon": "calendar"},
        "files": {"title": "Files", "icon": "files"},
        "folders": {"title": "Folders", "icon": "folders"},
        "redirects": {"title": "Redirects", "icon": "redirects"},
        "events": {"title": "Events", "icon": "events"},
        "users": {"title": "Users", "icon": "users"},
    }

    if object_name in page_details:
        details = page_details[object_name]
        return render_template(
            "list.html",
            title=details["title"],
            icon=details["icon"],
            object_name=object_name,
        )
    else:
        return "Page not found", 404


@app.route("/<object_name>/table")
@login_required
def view_table(object_name):
    page_details = {
        "companies": {"columns": ["name", "location", "website"], "model": Company},
        "contacts": {
            "columns": ["name", "title", "email", "phone", "company", "tags", "last_interaction_at"],
            "model": Contact,
        },
        "interactions": {
            "columns": [
                "notes_summary_one_line",
                "contact",
                "date",
                "opportunities",
                "tags",
            ],
            "model": Interaction,
        },
        "opportunities": {
            "columns": [
                "name",
                "date",
                "contact",
                "company",
                "notes_one_liner_summary",
                "value",
                "tags",
            ],
            "model": Opportunity,
        },
        "tasks": {
            "columns": [
                "name",
                "due_date",
                "status",
                "contact",
                "assigned_to",
                "interaction",
                "tags",
            ],
            "model": Task,
        },
        "tags": {"columns": ["name"], "model": Tag},
        "users": {
            "columns": [
                "name",
                "email",
                "telegram_id",
                "theme",
                "is_assistant",
                "assistant_system_prompt",
                "assistant_actions",
                "assistant_description",
                "assistant_example_prompts",
            ],
            "model": User,
        },
    }

    page = request.args.get("page", 1, type=int)
    per_page = 10
    model = page_details[object_name]["model"]

    # Start with the base query
    query = model.query

    # Iterate over request.args and apply filters that start with 'filter_'
    for arg_key in request.args:
        if arg_key.startswith("filter_"):
            # Extract the actual filter name by removing 'filter_' prefix
            filter_name = arg_key[7:]
            # Ensure the filter exists on the model to avoid security issues
            if hasattr(model, filter_name):
                filter_value = request.args.get(arg_key)
                # Apply the filter to the query
                query = query.filter(getattr(model, filter_name) == filter_value)

    objects_pagination = query.paginate(page=page, per_page=per_page, error_out=False)
    objects = objects_pagination.items

    return render_template(
        "table.html",
        objects=objects,
        object_name=object_name,
        pagination=objects_pagination,
        columns=page_details[object_name]["columns"],
        edit=f"{object_name}_edit",
        view=object_name + "_view",
        new=f"{object_name}_new",
    )


@app.route("/<object_name>/<int:id>", methods=["DELETE"])
@login_required
def delete_object(object_name, id):
    page_details = {
        "companies": {"model": Company},
        "contacts": {"model": Contact},
        "interactions": {"model": Interaction},
        "opportunities": {"model": Opportunity},
        "tasks": {"model": Task},
        "tags": {"model": Tag},
        "users": {"model": User},
    }
    item = page_details[object_name]["model"].query.get(id)
    if item:
        db.session.delete(item)
        db.session.commit()
    return ""


@app.route("/<object_name>/<int:id>/chat", methods=["GET"])
@login_required
def chat_object(object_name, id):
    page_details = {
        "companies": {"model": Company},
        "contacts": {"model": Contact},
        "interactions": {"model": Interaction},
        "opportunities": {"model": Opportunity},
        "tasks": {"model": Task},
        "tags": {"model": Tag},
        "users": {"model": User},
    }
    item = page_details[object_name]["model"].query.get(id)
    return ""


@app.route("/<object_name>/csv")
@login_required
def download_csv(object_name):
    page_details = {
        "companies": {"columns": ["name", "location", "website"], "model": Company},
        "contacts": {
            "columns": ["name", "title", "email", "phone", "company", "tags"],
            "model": Contact,
        },
        "interactions": {
            "columns": ["contact", "date", "notes", "opportunities", "tags"],
            "model": Interaction,
        },
        "opportunities": {
            "columns": ["contact", "date", "name", "notes", "value", "tags"],
            "model": Opportunity,
        },
        "tasks": {
            "columns": [
                "name",
                "due_date",
                "status",
                "contact",
                "assigned_to",
                "interaction",
                "tags",
            ],
            "model": Task,
        },
        "tags": {"columns": ["name", "color"], "model": Tag},
        "users": {
            "columns": [
                "name",
                "email",
                "telegram_id",
                "theme",
                "is_assistant",
                "assistant_system_prompt",
                "assistant_actions",
                "assistant_description",
                "assistant_example_prompts",
            ],
            "model": User,
        },
    }

    objects = page_details[object_name]["model"].query.all()

    csv = ""
    # add columns row
    csv += ";".join(page_details[object_name]["columns"]) + "\n"

    for object in objects:
        csv += (
            ";".join(
                [
                    str(getattr(object, column))
                    for column in page_details[object_name]["columns"]
                ]
            )
            + "\n"
        )
    return Response(
        csv,
        mimetype="text/csv",
        headers={"Content-disposition": f"attachment; filename={object_name}.csv"},
    )


@app.route("/contacts/<int:id>")
@login_required
def contacts_view(id):
    contact = Contact.query.get(id)
    return render_template("contact.html", contact=contact)


@app.route("/contacts/new", methods=["GET", "POST"])
@login_required
def contacts_new():
    form = ContactForm()
    if form.validate_on_submit():
        contact = Contact()
        contact.name = form.name.data
        contact.email = form.email.data
        contact.phone = form.phone.data
        contact.company = form.company.data
        contact.notes = strip_html_tags(form.notes.data)
        contact.upload_image(form.image.data)
        contact.tags = form.tags.data
        db.session.add(contact)
        db.session.commit()
        return redirect(url_for("view_list", object_name="contacts"))
    return render_template("form.html", form=form, url=url_for("contacts_new"))


@app.route("/contacts/<int:id>/edit", methods=["GET", "POST"])
@login_required
def contacts_edit(id):
    contact = Contact.query.get(id)
    form = ContactForm(obj=contact)
    if form.validate_on_submit():
        contact.name = form.name.data
        contact.email = form.email.data
        contact.phone = form.phone.data
        contact.company = form.company.data
        contact.notes = strip_html_tags(form.notes.data)
        contact.upload_image(form.image.data)
        contact.tags = form.tags.data
        db.session.commit()
        return redirect(url_for("view_list", object_name="contacts"))
    return render_template("form.html", form=form, url=url_for("contacts_edit", id=id))


@app.route("/companies/<int:id>")
@login_required
def companies_view(id):
    company = Company.query.get(id)
    return render_template("company.html", company=company)


@app.route("/companies/new", methods=["GET", "POST"])
@login_required
def companies_new():
    form = CompanyForm()
    if form.validate_on_submit():
        company = Company()
        company.name = form.name.data
        company.location = form.location.data
        company.website = form.website.data
        company.upload_image(form.image.data)
        db.session.add(company)
        db.session.commit()
        return redirect(url_for("view_list", object_name="companies"))
    return render_template("form.html", form=form, url=url_for("companies_new"))


@app.route("/companies/<int:id>/edit", methods=["GET", "POST"])
@login_required
def companies_edit(id):
    company = Company.query.get(id)
    form = CompanyForm(obj=company)
    if form.validate_on_submit():
        company.name = form.name.data
        company.location = form.location.data
        company.website = form.website.data
        company.upload_image(form.image.data)
        db.session.commit()
        return redirect(url_for("companies"))
    return render_template("form.html", form=form, url=url_for("companies_edit", id=id))


@app.route("/interactions/new", methods=["GET", "POST"])
@login_required
def interactions_new():
    form = InteractionForm()
    if form.validate_on_submit():
        interaction = Interaction()
        interaction.contact = form.contact.data
        interaction.date = form.date.data
        interaction.notes = strip_html_tags(form.notes.data)

        if interaction.notes:
            interaction.notes = strip_html_tags(form.notes.data)

            system_prompt = """\
Create a JSON object that provides a summary of a recent interaction with a client. This interaction could be in the form of an e-mail or a meeting transcript. The summary should be concise, focusing primarily on the process and factual aspects of the interaction. Be sure to include the names of the company and the project involved in the interaction. Additionally, the JSON should have keys for action points that emerged from this interaction. Each action point should be described in a one-liner format, detailing the specific task or follow-up action required. Here is an example of the JSON format:
{

    "summary": "Summary focusing on the process, facts, and mentioning company and project names.",
    "action_points": [
        "One-liner description of the first action point.", 
        "One-liner description of the second action point.",
        // Add more action points as necessary
        ]
    "oneliner_summary": "Provide a one-liner summary which can be used to describe the interaction."
}
Regardless of the input language, always translate to English and provide the JSON contents in English only. Only provide the JSON in your response, no additional text."""
            user_prompt = interaction.notes

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            response = query_openai(messages)

            result = json.loads(response["choices"][0]["message"]["content"])

            if result.get("action_points"):
                # create tasks for current_user
                for action_point in result["action_points"]:
                    task = Task()
                    task.name = action_point
                    task.due_date = datetime.utcnow()
                    task.status = "todo"
                    task.contact = interaction.contact
                    task.assigned_to = current_user
                    db.session.add(task)

                db.session.commit()

            interaction.notes_summary = result.get("summary")
            interaction.notes_summary_one_line = result.get("oneliner_summary")

        interaction.type_ = form.type_.data
        interaction.logged_by = current_user

        db.session.add(interaction)
        db.session.commit()
        return redirect(url_for("view_list", object_name="interactions"))
    return render_template("form.html", form=form, url=url_for("interactions_new"))


@app.route("/interactions/<int:id>/edit", methods=["GET", "POST"])
@login_required
def interactions_edit(id):
    interaction = Interaction.query.get(id)
    form = InteractionForm(obj=interaction)
    if form.validate_on_submit():
        interaction.contact = form.contact.data
        interaction.date = form.date.data

        if interaction.notes != strip_html_tags(form.notes.data):
            interaction.notes = strip_html_tags(form.notes.data)

            system_prompt = """\
        Create a JSON object that provides a summary of a recent interaction with a client. This interaction could be in the form of an e-mail or a meeting transcript. The summary should be concise, focusing primarily on the process and factual aspects of the interaction. Be sure to include the names of the company and the project involved in the interaction. Additionally, the JSON should have keys for action points that emerged from this interaction. Each action point should be described in a one-liner format, detailing the specific task or follow-up action required. Here is an example of the JSON format:
        {

            "summary": "Summary focusing on the process, facts, and mentioning company and project names.",
            "action_points": [
                "One-liner description of the first action point.", 
                "One-liner description of the second action point.",
                // Add more action points as necessary
                ]
            "oneliner_summary": "Provide a one-liner summary which can be used to describe the interaction."
        }
        Regardless of the input language, always translate to English and provide the JSON contents in English only. Only provide the JSON in your response, no additional text."""
            user_prompt = interaction.notes

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            response = query_openai(messages)

            result = json.loads(response["choices"][0]["message"]["content"])

            if result.get("action_points"):
                # create tasks for current_user
                for action_point in result["action_points"]:
                    task = Task()
                    task.name = action_point
                    task.due_date = datetime.utcnow()
                    task.status = "todo"
                    task.contact = interaction.contact
                    task.assigned_to = current_user
                    db.session.add(task)

                db.session.commit()

            interaction.notes_summary = result.get("summary")
            interaction.notes_summary_one_line = result.get("oneliner_summary")

        interaction.type_ = form.type_.data
        db.session.commit()
        return redirect(url_for("view_list", object_name="interactions"))
    return render_template(
        "form.html",
        form=form,
        url=url_for("interactions_edit", id=id),
    )


@app.route("/interactions/<int:id>")
@login_required
def interactions_view(id):
    interaction = Interaction.query.get(id)
    return render_template("interaction.html", interaction=interaction)


@app.route("/opportunity/<int:id>")
@login_required
def opportunities_view(id):
    opportunity = Opportunity.query.get(id)
    return render_template("opportunity.html", opportunity=opportunity)


@app.route("/opportunities/new", methods=["GET", "POST"])
@login_required
def opportunities_new():
    form = OpportunityForm()
    if form.validate_on_submit():
        opportunity = Opportunity()
        opportunity.contact = form.contact.data
        opportunity.date = form.date.data
        opportunity.notes = form.notes.data
        opportunity.value = form.value.data
        opportunity.name = form.name.data
        db.session.add(opportunity)
        db.session.commit()
        return redirect(url_for("view_list", object_name="opportunities"))
    return render_template("form.html", form=form, url=url_for("opportunities_new"))


@app.route("/opportunities/<int:id>/edit", methods=["GET", "POST"])
@login_required
def opportunities_edit(id):
    opportunity = Opportunity.query.get(id)
    form = OpportunityForm(obj=opportunity)
    if form.validate_on_submit():
        opportunity.contact = form.contact.data
        opportunity.date = form.date.data
        opportunity.notes = form.notes.data
        opportunity.value = form.value.data
        opportunity.name = form.name.data
        db.session.commit()
        return redirect(url_for("opportunities"))
    return render_template(
        "form.html",
        form=form,
        url=url_for("edit_opportunity", opportunity_id=id),
    )


@app.route("/tasks/new", methods=["GET", "POST"])
@login_required
def tasks_new():
    form = TaskForm()
    if form.validate_on_submit():
        task = Task()
        task.name = form.name.data
        task.due_date = form.due_date.data
        task.status = form.status.data
        task.contact = form.contact.data
        task.assigned_to = form.assigned_to.data
        task.tags = form.tags.data
        db.session.add(task)
        db.session.commit()

        return redirect(url_for("view_list", object_name="tasks"))
    return render_template("form.html", form=form, url=url_for("tasks_new"))


@app.route("/tasks/<int:id>/edit", methods=["GET", "POST"])
@login_required
def tasks_edit(id):
    task = Task.query.get(id)
    form = TaskForm(obj=task)
    if form.validate_on_submit():
        task.name = form.name.data
        task.due_date = form.due_date.data
        task.status = form.status.data
        task.contact = form.contact.data
        task.assigned_to = form.assigned_to.data
        task.tags = form.tags.data
        db.session.commit()
        return redirect(url_for("view_list", object_name="tasks"))
    return render_template("form.html", form=form, url=url_for("tasks_edit", id=id))


def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(
                    1 + min((distances[i1], distances[i1 + 1], distances_[-1]))
                )
        distances = distances_
    return distances[-1]


def custom_match_score(target, candidate):
    distance = levenshtein_distance(target, candidate)
    # Decrease distance for partial matches (substring presence)
    if target in candidate:
        distance -= len(target) / 2  # Adjust the factor as needed
    return distance


def find_best_matches(target, candidates, n):
    scores = [
        (candidate, custom_match_score(target, candidate)) for candidate in candidates
    ]
    scores.sort(key=lambda x: x[1])
    return scores[:n]


def new_task_action(description, name):
    """
    Create a new task

    Arguments:
        description {string} -- Description of the task
    """
    users = User.query.all()

    # find closest matching user name
    closest_matches = find_best_matches(name, [user.name for user in users], 3)
    print("closest matches:" + str(closest_matches))
    if closest_matches:
        user = User.query.filter_by(name=closest_matches[0][0]).first()

    task = Task()
    task.name = description

    if user:
        task.assigned_to = user

    db.session.add(task)
    db.session.commit()
    return task


@app.route("/tasks/<int:id>")
@login_required
def tasks_view(id):
    task = Task.query.get(id)
    return render_template("task.html", task=task)


@app.route("/users/<int:id>")
@login_required
def users_view(id):
    user = User.query.get(id)
    return render_template("user.html", user=user)


@app.route("/users/new", methods=["GET", "POST"])
@login_required
def users_new():
    form = UserForm()
    if form.validate_on_submit():
        user = User()
        user.name = form.name.data
        user.email = form.email.data
        user.set_password(form.password.data)
        user.telegram_id = form.telegram_id.data
        user.theme = form.theme.data
        user.is_assistant = form.is_assistant.data
        user.assistant_system_prompt = form.assistant_system_prompt.data
        user.assistant_actions = form.assistant_actions.data
        user.assistant_description = form.assistant_description.data
        user.assistant_example_prompts = form.assistant_example_prompts.data

        db.session.add(user)
        db.session.commit()
        return redirect(url_for("users"))
    return render_template("form.html", form=form, url=url_for("users_new"))


@app.route("/users/<int:id>/edit", methods=["GET", "POST"])
@login_required
def users_edit(id):
    user = User.query.get(id)
    form = UserForm(obj=user)

    if form.validate_on_submit():
        user.name = form.name.data
        user.email = form.email.data
        user.telegram_id = form.telegram_id.data
        if form.password.data:
            user.set_password(form.password.data)
        user.theme = form.theme.data
        user.is_assistant = form.is_assistant.data
        user.assistant_system_prompt = form.assistant_system_prompt.data
        user.assistant_actions = form.assistant_actions.data
        user.assistant_description = form.assistant_description.data
        user.assistant_example_prompts = form.assistant_example_prompts.data
        user.upload_image(request.files["image"])

        db.session.commit()
        return redirect(url_for("view_list", object_name="users"))
    return render_template("form.html", form=form, url=url_for("users_edit", id=id))


@app.route("/notifications")
@login_required
def notifications():
    notifications = Notification.query.filter_by(user=current_user)

    for notification in notifications:
        notification.read = True
        db.session.commit()

    return render_template("notifications.html", notifications=notifications)


@app.route("/tags/new", methods=["GET", "POST"])
@login_required
def tags_new():
    form = TagForm()
    if form.validate_on_submit():
        tag = Tag()
        tag.name = form.name.data
        tag.color = form.color.data.replace("#", "")
        db.session.add(tag)
        db.session.commit()
        return redirect(url_for("view_list", object_name="tags"))
    return render_template("form.html", form=form, url=url_for("tags_new"))


@app.route("/tags/<int:id>/edit", methods=["GET", "POST"])
@login_required
def tags_edit(id):
    tag = Tag.query.get(id)
    form = TagForm(obj=tag)
    if form.validate_on_submit():
        tag.name = form.name.data
        tag.color = form.color.data.replace("#", "")
        db.session.commit()
        return redirect(url_for("view_list", object_name="tags"))
    return render_template("form.html", form=form, url=url_for("tags_edit", id=id))


@app.route("/tag/<int:id>")
@login_required
def tags_view(id):
    tag = Tag.query.get(id)
    return render_template("tag.html", tag=tag)


@app.route("/chat/new/<int:user_id>")
@login_required
def new_chat(user_id):
    conversation = Conversation()
    conversation.participants.append(current_user)

    user = User.query.get(user_id)
    conversation.participants.append(user)

    db.session.add(conversation)
    db.session.commit()

    print(conversation.id)
    return redirect(url_for("chat_conversation", id=conversation.id))


@app.route("/documents", methods=["GET", "POST"], defaults={"id": None})
@app.route("/documents/folder/<int:id>", methods=["GET", "POST"])
@login_required
def documents(id):
    folders = Folder.query.filter_by(parent_id=id if id else None).all()

    if not id:
        files = File.query.filter_by(folder_id=None).all()
    else:
        files = File.query.filter_by(folder_id=id).all()

    if request.method == "POST":
        file_ = request.files["file"]

        if id:
            print(id)
            folder = Folder.query.get(id)
            print(folder)
        else:
            folder = None

        filename = secure_filename(file_.filename)
        file_.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))

        file_ = File()
        file_.original_filename = filename
        file_.filename = filename
        # determine file type
        if filename.endswith(".pdf"):
            file_.file_type = "pdf"
        elif filename.endswith(".txt"):
            file_.file_type = "txt"
        elif filename.endswith(".wav") or filename.endswith(".mp3"):
            file_.file_type = "audio"
        elif filename.endswith(".jpg") or filename.endswith(".png"):
            file_.file_type = "image"
        else:
            return abort(400)

        file_.md5 = hashlib.md5(
            open(os.path.join(app.config["UPLOAD_FOLDER"], filename), "rb").read()
        ).hexdigest()
        file_.size = os.path.getsize(
            os.path.join(app.config["UPLOAD_FOLDER"], filename)
        )
        file_.folder_id = folder.id if folder else None
        file_.uploaded_by = current_user

        # extensions to create embeddings from:
        # txt, pdf
        if filename.endswith(".txt"):
            file_.status = "processing"
            file_.text = open(
                os.path.join(app.config["UPLOAD_FOLDER"], filename), "r"
            ).read()
            db.session.add(file_)
            db.session.commit()

        elif filename.endswith(".pdf"):
            file_.status = "processing"
            db.session.add(file_)
            db.session.commit()
            db.session.flush()

            global p
            p = Process(target=process_file, args=(file_.id,))
            p.start()

        else:
            file_.status = "processed"
            db.session.add(file_)
            db.session.commit()

    print(files)
    return render_template(
        "documents.html",
        folders=folders,
        files=files,
        current_folder_id=id if id else None,
    )


@app.route("/documents/folder/new", methods=["GET", "POST"], defaults={"id": None})
@app.route("/documents/folder/<int:id>/new", methods=["GET", "POST"])
@login_required
def new_folder(id):
    form = FolderForm()
    if form.validate_on_submit():
        folder = Folder()
        folder.name = form.name.data
        folder.parent_id = id
        db.session.add(folder)
        db.session.commit()
        return redirect(url_for("documents"))
    return render_template("form.html", form=form, url=url_for("new_folder"))


@app.route("/documents/file/<int:id>")
@login_required
def document(id):
    file_ = File.query.get(id)
    # return file as response
    print(file_.filename)
    return send_from_directory(app.config["UPLOAD_FOLDER"], file_.filename)

@app.route("/documents/file")
@login_required
def file():
    secret = request.args.get("secret")

    print(secret)

    if not secret:
        return abort(404)

    file_secret = FileSecret.query.filter_by(secret=secret).first()
    print(file_secret)

    if not file_secret:
        return abort(403)

    if file_secret.expires_at < datetime.utcnow():
        return abort(401)

    if file_secret.file.filename.endswith(".html"):
        return send_from_directory(
            app.config["UPLOAD_FOLDER"], file_secret.file.filename, as_attachment=False
        )
    else:
        return send_from_directory(
            app.config["UPLOAD_FOLDER"], file_secret.file.filename, as_attachment=True
        )


@app.route("/documents/<int:id>")
@login_required
def document_folder(id):
    file = File.query.get(id)
    return render_template("document.html", file=file)


@app.route("/documents/<int:id>/status")
@login_required
def document_status(id):
    file = File.query.get(id)
    if file.status == "processed":
        return file.status, 286
    return render_template("processing_status.html")


@app.route("/files/table")
@login_required
def files_table():
    files = File.query.all()

    # if none of the files have 'processing' as status, return 16
    if not any([file.status == "processing" for file in files]):
        response_code = 286
    else:
        response_code = 200

    # paginate
    page = request.args.get("page", 1, type=int)
    per_page = 10
    pagination = File.query.paginate(page=page, per_page=per_page, error_out=False)

    files = pagination.items

    return (
        render_template(
            "table.html",
            object_name="files",
            pagination=pagination,
            objects=files,
            columns=["filename", "size", "uploaded_by", "status", "tags"],
            delete="delete_file",
            view="document",
        ),
        response_code,
    )


@app.route("/files/new")
@login_required
def files_new():
    pass


@app.route("/files/<int:id>", methods=["DELETE"])
@login_required
def files_delete(id):
    file = File.query.get(id)
    if file:
        db.session.delete(file)
        db.session.commit()
    return ""


@app.route("/documents/file/<int:id>/share", methods=["GET", "POST"])
@login_required
def share_file(id):
    form = ShareFileForm()
    if form.validate_on_submit():
        file_ = File.query.get(id)
        file_secret = FileSecret()
        file_secret.file = file_
        file_secret.contact = form.contact.data
        file_secret.expires_at = form.expires_at.data
        file_secret.update_secret()
        db.session.add(file_secret)
        db.session.commit()
        return redirect(url_for("documents"))
    return render_template("form.html", form=form, url=url_for("share_file", id=id))


@app.route("/shared-files")
@login_required
def shared_documents():
    return render_template("shared_files.html")

@app.route("/assistant/<int:id>/chat")
@login_required
def assistant_chat(id):
    assistant = User.query.get(id)

    conversation = Conversation()

    if current_user not in conversation.participants:
        conversation.participants.append(current_user)

    if assistant not in conversation.participants:
        conversation.participants.append(assistant)

    db.session.add(conversation)
    db.session.commit()

    return render_template(
        "assistant_chat.html", assistant=assistant, conversation=conversation
    )


def authenticated_only(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        if not current_user.is_authenticated:
            disconnect()
        else:
            return f(*args, **kwargs)

    return wrapped


@socketio.on("connect")
@authenticated_only
def handle_connect():
    print("Client connected")


@socketio.on("disconnect")
@authenticated_only
def handle_disconnect():
    print("Client disconnected")


def query_openai_stream(messages):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
    }
    data = {"model": "gpt-4-turbo-preview", "messages": messages, "stream": True}

    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        json=data,
        headers=headers,
        stream=True,
    )

    for line in response.iter_lines():
        if line:
            decoded_line = line.decode("utf-8")
            if decoded_line == "data: [DONE]":
                break

            json_line = json.loads(decoded_line[len("data: ") :])
            if json_line.get("choices") and json_line["choices"][0].get("delta"):
                print(json_line["choices"][0]["delta"]["content"])
                yield json_line["choices"][0]["delta"]["content"]


streams_active = {}


@socketio.on("send_message")
@authenticated_only
def handle_message(data):
    conversation_id = data.get("conversation_id")
    conversation = Conversation.query.get(conversation_id)

    if current_user not in conversation.participants:
        return

    message = Message()
    message.sender = current_user
    message.conversation = conversation
    message.text = data.get("text")
    db.session.add(message)

    db.session.commit()

    data["sender"] = current_user.name

    socketio.emit("receive_message", data)

    # if the other participant is an assistant, send the message to openai
    print(conversation.participants)
    for u in conversation.participants:
        if u.is_assistant:
            assistant = u
            print(assistant)
            break

    if assistant:
        # get last 5 messages from conversation
        messages = [{"role": "system", "content": assistant.assistant_system_prompt}]

        # document assistant
        if assistant.id == 99999:
            embeddings = Embedding.query.all()

            # convert to numpy array
            embedding_vectors = np.array([e.vector for e in embeddings])

            # get embedding for user message
            user_embedding = get_openai_embedding(message.text)

            def cosine_similarity(mat, query, top_k):
                # Normalize the matrices
                mat_norm = mat / np.linalg.norm(mat, axis=1, keepdims=True)
                query_norm = query / np.linalg.norm(query)

                # Compute cosine similarity
                similarity = np.dot(mat_norm, query_norm)

                # Get top k indices
                top_k_indices = np.argsort(similarity)[-top_k:][::-1]

                return top_k_indices, similarity[top_k_indices]

            top_k_indices, similarity = cosine_similarity(
                embedding_vectors, user_embedding, 3
            )
            matched_chunks = [embeddings[i].chunk for i in top_k_indices]
            matched_chunks_texts = [chunk.text for chunk in matched_chunks]

            user_prompt = f"Question: {message.text}\n\nContext: {' '.join(matched_chunks_texts)}\n\n"

            messages.append({"role": "user", "content": user_prompt})

        else:
            for message in conversation.messages[-5:]:
                messages.append(
                    {
                        "role": (
                            "user" if message.sender == current_user else "assistant"
                        ),
                        "content": message.text,
                    }
                )

        m = Message()
        m.sender = assistant
        m.conversation = conversation
        m.text = ""
        db.session.add(m)
        db.session.commit()

        db.session.flush()
        socketio.emit(
            "receive_message", {"sender": assistant.name, "text": "", "uid": m.id}
        )

        # add stream to streams_active for this conversation
        streams_active[conversation.id] = True

        for response_message in query_openai_stream(messages):
            if streams_active[conversation.id]:
                m.text += response_message
                db.session.commit()

                socketio.emit(
                    "receive_message_update",
                    {"uid": m.id, "text": response_message, "operation": "append"},
                )
            else:
                break

        if assistant.id == 99999:
            # question, answer, first chunk
            # prompt: "This is the answer to this question this is the source of the answer. Give me a literal sentence from the source to highlight the context in the source. Only give the exact sentence, nothing else"

            system_prompt = "The user will give you (a) a question, (b) an answer and (c) and a few extracts from the source document. You need to identify the source of the answer and provide a literal sentence from the source to highlight the context in the source. Only give the exact copy of the snippet, nothing else."

            # query last answer from the assistant
            answer = (
                Message.query.filter_by(conversation=conversation, sender=assistant)
                .order_by(Message.created_at.desc())
                .first()
            )

            user_prompt = f"Question: {message.text} ?\n\nAnswer: {answer.text}\n\nContext: {matched_chunks[1].text}"
            result = query_openai(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            )

            answer_text = result["choices"][0]["message"]["content"]

            def find_overlap(s1, s2):
                if len(s1) > len(s2):
                    s1, s2 = s2, s1  # Ensure s1 is the shorter string for efficiency
                max_overlap = ""
                for i in range(len(s1)):
                    for j in range(i + 1, len(s1) + 1):
                        if s1[i:j] in s2 and len(s1[i:j]) > len(max_overlap):
                            max_overlap = s1[i:j]
                return max_overlap

            text_to_highlight_in_chunk = find_overlap(
                answer_text, matched_chunks[1].text
            )

            # highlighted_text = matched_chunks[1].text.replace(text_to_highlight_in_chunk, f"<b>{text_to_highlight_in_chunk}</b>")

            # socketio.emit('receive_message_update', {'uid': m.id, 'text': highlighted_text, 'operation': 'append_html'})

            # identify the files for the matched chunks
            files = []
            for chunk in matched_chunks:
                if chunk.file not in files:
                    files.append(chunk.file)

            html = render_template(
                "attachments.html",
                files=files,
                highlighted_text=text_to_highlight_in_chunk,
            )

            socketio.emit(
                "receive_message_update",
                {"uid": m.id, "text": html, "operation": "append_html"},
            )

        socketio.emit(
            "receive_message_update", {"uid": m.id, "text": "", "operation": "done"}
        )


@socketio.on("stop_assistant")
@authenticated_only
def handle_stop_assistant(data):
    conversation_id = data.get("conversation_id")
    conversation = Conversation.query.get(conversation_id)

    if current_user not in conversation.participants:
        return

    for u in conversation.participants:
        if u.is_assistant:
            assistant = u
            print(assistant)
            break

    if assistant:
        streams_active[conversation.id] = False


@app.route("/search", methods=["GET", "POST"])
@login_required
def search():
    if request.method == "POST":
        if request.form.get("query"):
            query = request.form["query"]
        else:
            return

        contacts = Contact.query.filter(Contact.name.ilike(f"%{query}%")).all()
        companies = Company.query.filter(Company.name.ilike(f"%{query}%")).all()
        tasks = Task.query.filter(Task.name.ilike(f"%{query}%")).all()
        interactions = Interaction.query.filter(
            Interaction.notes.ilike(f"%{query}%")
        ).all()
        opportunities = Opportunity.query.filter(
            Opportunity.notes.ilike(f"%{query}%")
        ).all()
        tags = Tag.query.filter(Tag.name.ilike(f"%{query}%")).all()
        users = User.query.filter(User.name.ilike(f"%{query}%")).all()
        files = File.query.filter(File.filename.ilike(f"%{query}%")).all()

        return render_template(
            "search_results.html",
            contacts=contacts,
            companies=companies,
            tasks=tasks,
            interactions=interactions,
            opportunities=opportunities,
            tags=tags,
            users=users,
            query=query,
            files=files,
        )

    return render_template("search.html")


@app.route("/uploads/<path:filename>")
@login_required
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/timeline")
@login_required
def timeline():
    interactions = Interaction.query.order_by(Interaction.date.desc()).all()
    return render_template("timeline.html", interactions=interactions)
