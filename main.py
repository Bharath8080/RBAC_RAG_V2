import os
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
from typing import Dict 
from pymongo import MongoClient
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

load_dotenv()

# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")

# Initialize MongoDB client
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
users_collection = db["users"]


ROLE_PERMISSIONS = {
    "finance": ["finance"],
    "marketing": ["marketing"],
    "hr": ["hr"],
    "engineering": ["engineering"],
    "employee": ["general"]
}


security = HTTPBasic()
app = FastAPI()

class Query(BaseModel):
    message: str
    role: str

def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    username = credentials.username
    password = credentials.password
    
    # Find user in MongoDB
    user = users_collection.find_one({"username": username})
    
    if not user or user["password"] != password:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    
    return {"username": username, "role": user["role"]}

@app.get("/login")
def login(user=Depends(authenticate)):
    return {"message": f"Welcome {user['username']}!", "role": user["role"]}


@app.post("/chat") 
def chat_with_docs(query_data: Query, user=Depends(authenticate)): 
    user_actual_role = user['role']
    # Convert the department name to lowercase to match the vector store folders
    target_department_role = query_data.role.lower() 
    
    # Log access attempt for security
    print(f"🔐 Access attempt: User '{user['username']}' (Role: {user_actual_role}) trying to access '{target_department_role}' department")
    
    # Check RBAC permissions
    user_permissions = ROLE_PERMISSIONS.get(user_actual_role, [])
    if target_department_role not in user_permissions:
        print(f"❌ Access denied: {user_actual_role} users cannot access {target_department_role} data")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"🚫 Access Denied: {user_actual_role.capitalize()} users cannot access {target_department_role.capitalize()} data. Your role only has access to: {', '.join(user_permissions).capitalize()}"
        )
    
    print(f"✅ Access granted: {user_actual_role} user accessing {target_department_role} data")

    # Pre-validate the question for cross-department access attempts
    question_lower = query_data.message.lower()
    unauthorized_departments = ["finance", "marketing", "engineering", "general"]
    
    # Check if user is asking about other departments
    for dept in unauthorized_departments:
        if dept in question_lower and dept != target_department_role:
            print(f"🚫 Cross-department access attempt detected: {user_actual_role} user asking about {dept} department")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"🚫 Access Denied: You are asking about {dept.capitalize()} department data. As a {user_actual_role.capitalize()} user, you can only access {target_department_role.capitalize()} department data."
            )
    
    # Check for company-wide questions that might be unauthorized
    company_wide_keywords = ["executive summary", "company summary", "overall", "company-wide", "all departments"]
    for keyword in company_wide_keywords:
        if keyword in question_lower:
            print(f"🚫 Company-wide access attempt detected: {user_actual_role} user asking about {keyword}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"🚫 Access Denied: You are asking for company-wide information ({keyword}). As a {user_actual_role.capitalize()} user, you can only access {target_department_role.capitalize()} department data."
            )

    vector_store_path = os.path.join("vector_store", target_department_role)

    if not os.path.exists(vector_store_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Knowledge base not found for department: {target_department_role.capitalize()}."
        )

    try:
        db = FAISS.load_local(
            folder_path=vector_store_path,
            embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
            allow_dangerous_deserialization=True 
        )
        retriever = db.as_retriever()
        docs = retriever.invoke(query_data.message)
        if not docs:
            return {"response": "I couldn't find relevant information in the knowledge base for your query."}
        # Simple prompt for direct answers
        prompt = ChatPromptTemplate.from_template(
            """You are a helpful assistant. Answer the question based on the provided context.

Context:
{context}

Question: {input}

Answer:"""
        )
        llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"), temperature=0.1)

        chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
        response = chain.invoke({
            "input": query_data.message, 
            "context": docs,
            "department": target_department_role.capitalize()
        })
        return {"response": response}

    except Exception as e:
        import traceback
        print(f"An error occurred in /chat endpoint: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Internal server error: {str(e)}")

@app.get("/")
def root():
    return {"message": "FinSolve RBAC Chatbot API is running."}

@app.get("/test")
def test_authentication(user=Depends(authenticate)):
    return {"message": f"Hello {user['username']}! Your role is {user['role']}.", "role": user["role"]}

@app.get("/my-permissions")
def get_user_permissions(user=Depends(authenticate)):
    """Get the user's role and department access permissions"""
    user_role = user['role']
    user_permissions = ROLE_PERMISSIONS.get(user_role, [])
    
    return {
        "username": user['username'],
        "role": user_role,
        "department_access": user_permissions,
        "message": f"User {user['username']} has {user_role} role with access to: {', '.join(user_permissions)}"
    }

@app.get("/rbac-test")
def test_rbac_access(user=Depends(authenticate)):
    """Test RBAC access for different departments"""
    user_role = user['role']
    user_permissions = ROLE_PERMISSIONS.get(user_role, [])
    
    test_results = {}
    for dept in ["finance", "marketing", "hr", "engineering", "general"]:
        test_results[dept] = {
            "accessible": dept in user_permissions,
            "reason": "Authorized" if dept in user_permissions else "Not authorized for this role"
        }
    
    return {
        "username": user['username'],
        "role": user_role,
        "authorized_departments": user_permissions,
        "access_test_results": test_results,
        "message": f"RBAC test completed for {user['username']} ({user_role})"
    }

@app.get("/devops-engineers")
def get_devops_engineers(user=Depends(authenticate)):
    """Get all DevOps engineers from the HR data"""
    import pandas as pd
    
    # Check if user has access to HR data
    if user['role'] not in ['hr', 'employee']:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access HR data."
        )
    
    try:
        # Read the HR data
        hr_data_path = os.path.join("data", "hr", "hr_data.csv")
        if not os.path.exists(hr_data_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="HR data file not found."
            )
        
        df = pd.read_csv(hr_data_path)
        
        # Filter DevOps engineers
        devops_engineers = df[df['role'] == 'DevOps Engineer']
        
        if devops_engineers.empty:
            return {"message": "No DevOps engineers found.", "count": 0, "engineers": []}
        
        # Convert to list of dictionaries
        engineers_list = devops_engineers.to_dict('records')
        
        return {
            "message": f"Found {len(engineers_list)} DevOps engineer(s)",
            "count": len(engineers_list),
            "engineers": engineers_list
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error reading HR data: {str(e)}"
        )

@app.get("/risk-analysts")
def get_risk_analysts(user=Depends(authenticate)):
    """Get all Risk Analysts from the HR data"""
    import pandas as pd
    
    # Check if user has access to HR data
    if user['role'] not in ['hr', 'employee']:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access HR data."
        )
    
    try:
        # Read the HR data
        hr_data_path = os.path.join("data", "hr", "hr_data.csv")
        if not os.path.exists(hr_data_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="HR data file not found."
            )
        
        df = pd.read_csv(hr_data_path)
        
        # Filter Risk Analysts
        risk_analysts = df[df['role'] == 'Risk Analyst']
        
        if risk_analysts.empty:
            return {"message": "No Risk Analysts found.", "count": 0, "analysts": []}
        
        # Convert to list of dictionaries
        analysts_list = risk_analysts.to_dict('records')
        
        return {
            "message": f"Found {len(analysts_list)} Risk Analyst(s)",
            "count": len(analysts_list),
            "analysts": analysts_list
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error reading HR data: {str(e)}"
        )

@app.get("/verify-vector-store-access")
def verify_vector_store_access(user=Depends(authenticate)):
    """Verify that the user can only access their authorized vector store"""
    user_role = user['role']
    user_permissions = ROLE_PERMISSIONS.get(user_role, [])
    
    verification_results = {}
    
    for dept in ["finance", "marketing", "hr", "engineering", "general"]:
        vector_store_path = os.path.join("vector_store", dept)
        is_authorized = dept in user_permissions
        path_exists = os.path.exists(vector_store_path)
        
        verification_results[dept] = {
            "authorized": is_authorized,
            "vector_store_path": vector_store_path,
            "path_exists": path_exists,
            "access_granted": is_authorized and path_exists,
            "message": f"{'✅ Authorized' if is_authorized else '❌ Not authorized'} - Path: {vector_store_path} {'(exists)' if path_exists else '(missing)'}"
        }
    
    return {
        "username": user['username'],
        "role": user_role,
        "authorized_departments": user_permissions,
        "verification_results": verification_results,
        "message": f"Vector store access verification completed for {user['username']} ({user_role})"
    }