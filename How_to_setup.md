# Setup and Contribution Guide

This guide outlines the steps to run, develop, and contribute to the **Multi-Agent Emergent Communication Control System** application.

---

## 1. Prerequisites

Before you start, make sure you have the following installed on your machine:
*   **Git**: For version control.
*   **Docker Desktop**: Required to run the backend and frontend containers. **Ensure Docker is running** before executing any commands.
*   **NVIDIA Container Toolkit** (Optional but Recommended): Required if you plan to train policies using the GPU. If you don't have it, PyTorch will fall back to CPU training.

---

## 2. Running the Application

Follow these steps to run the application locally:

### Step 1: Clone the Repository
Clone the repository to your local machine and navigate into the project root:
```bash
git clone <repository-url>
cd MAC_COPY
```

### Step 2: Ensure Docker is Running
Launch **Docker Desktop** (on Windows/macOS) or ensure the docker daemon is running (on Linux).

### Step 3: Spin Up the Containers
Build and run the frontend and backend services using Docker Compose:
```bash
docker-compose up --build
```
This will:
*   Build the PyTorch backend container and expose it on port `8000`.
*   Build the React (Vite) frontend container and expose it on port `5173`.
*   Establish volume mounts so changes to your local files are instantly synced inside the running container workspace.

### Step 4: Access the Dashboard
Once the services are active, open your web browser and go to:
*   **Frontend UI**: [http://localhost:5173](http://localhost:5173)
*   **Backend API Docs (Swagger)**: [http://localhost:8000/docs](http://localhost:8000/docs)

### Step 5: (Optional) Generate Rollout GIFs
To view agent rollouts visually:
1. Run/select a completed run in the **Your Runs Library** list.
2. Click **Generate Rollout GIF** in the visualizer card. The backend will compile the simulation on a virtual headless display (`xvfb`) and load the animation on your dashboard.

---

## 3. Git Contribution Workflow

To contribute changes to the project, follow this structured Git workflow:

### Step 1: Keep Your Local Repository Synchronized
Before starting any new work, switch to the base development branch (e.g., `main` or the main developer branch) and pull the latest updates to avoid merge conflicts:
```bash
# Switch to the main branch
git checkout main

# Fetch and merge the latest changes from GitHub
git pull origin main
```

### Step 2: Create a New Feature Branch
Create a new branch for your specific feature or fix. Use a descriptive name:
```bash
git checkout -b <your-name>/<feature-or-fix-description>
# Example: git checkout -b anurag/add-degradation-detection
```

### Step 3: Implement Your Changes
Make edits to your code. Test them locally by running the Docker containers to make sure the app builds and functions correctly.

### Step 4: Stage and Commit Your Changes
Check your modified files, stage them, and write a descriptive commit message:
```bash
# Check modified files
git status

# Stage all modified files (or specify individual files)
git add .

# Commit with a meaningful message
git commit -m "feat: added online degradation detector to mpe runner"
```

### Step 5: Push Your Branch to GitHub
Push your local branch to the remote GitHub repository:
```bash
git push -u origin <your-branch-name>
# Example: git push -u origin anurag/add-degradation-detection
```

### Step 6: Create a Pull Request (PR)
1. Go to the repository page on GitHub.
2. You will see a banner prompting you to open a Pull Request for your recently pushed branch.
3. Click **Compare & pull request**.
4. Describe your changes and request a review from your team members.
