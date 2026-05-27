# Comprehensive Red Hat EC2 Deployment Guide

This document outlines the step-by-step process for deploying the RAG Chatbot onto a fresh Red Hat Enterprise Linux (RHEL) EC2 instance on AWS. It covers system preparation, dependency installation, security configurations, and application launch.

---

## Prerequisites

1. **AWS EC2 Instance**: A running EC2 instance using the Red Hat Enterprise Linux AMI.
2. **Key Pair (Optional)**: The `.pem` or `.ppk` file used to SSH into your instance. (Not required if using AWS SSM).
3. **Security Group**: Ensure your EC2 Security Group allows inbound traffic on:
   - **Port 22** (SSH) from your IP. (Not required if using AWS SSM).
   - **Port 80** (HTTP) from Anywhere (`0.0.0.0/0`).

---

## Step 1: Connect to your EC2 Instance

### Option A: AWS Systems Manager (SSM) Session Manager (Recommended)
This is the most secure method, requiring no open SSH ports or `.pem` keys.
1. Open the AWS EC2 Console.
2. Select your instance and click **Connect**.
3. Choose the **Session Manager** tab and click **Connect** to open a browser-based terminal.
4. Once connected, run the following command to switch to the standard user account:
   ```bash
   sudo su - ec2-user
   ```

### Option B: Traditional SSH
Open your local terminal (or PowerShell) and connect to the instance via SSH.
```bash
ssh -i /path/to/your-key.pem ec2-user@<your-ec2-public-ip>
```

---

## Step 2: Update System & Install Core Dependencies

Red Hat uses the `dnf` package manager. We need to install Python, Git, Nginx, and essential C++ compilers required for building vector libraries like `faiss` and `llama-cpp-python`.

```bash
# 1. Update all existing system packages
sudo dnf update -y

# 2. Install Python 3.11/3.12, pip, Git, and Nginx
sudo dnf install -y python3 python3-pip python3-devel git nginx

# 3. Install GCC and Development Tools (Crucial for AI packages)
sudo dnf groupinstall -y "Development Tools"
```

---

## Step 3: Transfer the Project Files

You need to get your local project files onto the EC2 instance.

### Option A: Using Git (Highly Recommended)
Regardless of how you connected (SSM or SSH), the easiest way is to clone your repository from GitHub/GitLab:
```bash
git clone <your-repository-url>
cd <your-repository-folder>
```

### Option B: Using S3 (If using SSM and no Git)
Upload your project as a `.zip` file to an AWS S3 bucket, then download it from your SSM terminal:
```bash
aws s3 cp s3://your-bucket-name/project.zip .
unzip project.zip
cd project
```

### Option C: Using SCP (If using Traditional SSH)
If your code is purely local and you connected via SSH, open a **new** terminal on your Windows machine and run:
```powershell
scp -i \path\to\your-key.pem -r "c:\Users\Harsh\HiHarsh\Coding\Python\RAG Closed context QA" ec2-user@<your-ec2-public-ip>:/home/ec2-user/RAG_Chatbot
```
Then, back on your EC2 SSH session:
```bash
cd /home/ec2-user/RAG_Chatbot
```

---

## Step 4: Python Virtual Environment Setup

Always isolate Python dependencies to avoid breaking system tools.

```bash
# Create a virtual environment named 'venv'
python3 -m venv venv

# Activate the virtual environment
source venv/bin/activate
```
*(Your terminal prompt should now be prefixed with `(venv)`).*

---

## Step 5: Install Python Dependencies

With the virtual environment active, install all the required AI, UI, and backend packages.

```bash
# Upgrade pip to the latest version to avoid build errors
pip install --upgrade pip

# Install dependencies from the requirements file
pip install -r requirements.txt
```

> **Note**: Building `llama-cpp-python` and `faiss-cpu` might take a few minutes as they are compiled from C++ source code during installation.

---

## Step 6: Configure Nginx (Reverse Proxy)

We use Nginx to securely route standard web traffic from port 80 to the internal Streamlit server running on port 8501, specifically handling the `/askbot` path.

```bash
# 1. Copy the custom askbot Nginx configuration
sudo cp deploy/nginx_askbot.conf /etc/nginx/conf.d/askbot.conf

# 2. Check for conflicts with the default Nginx config. 
# Red Hat's default /etc/nginx/nginx.conf listens on port 80.
# You can safely comment out the default server block in that file using `sudo nano /etc/nginx/nginx.conf` 
# or completely remove the default block to prevent port collisions.

# 3. Enable Nginx to start on system boot
sudo systemctl enable nginx

# 4. Restart Nginx to apply changes
sudo systemctl restart nginx
```

---

## Step 7: Configure Red Hat Security (SELinux & Firewall)

Red Hat has strict security policies (SELinux and firewalld) enabled by default. If you skip this step, Nginx will return a `502 Bad Gateway` error because it is blocked from routing traffic locally.

```bash
# 1. Allow Nginx to make local network connections (Crucial for Reverse Proxy)
sudo setsebool -P httpd_can_network_connect 1

# 2. Open Port 80 in the system firewall
sudo firewall-cmd --permanent --add-service=http
sudo firewall-cmd --reload
```

---

## Step 8: Launch the Chatbot Application

Finally, start the Streamlit application in the background. We provided a `start_app.sh` script to handle this via `nohup`.

```bash
# Make the deployment script executable
chmod +x deploy/start_app.sh

# Run the deployment script
./deploy/start_app.sh
```

You can monitor the application logs at any time by viewing the output file:
```bash
tail -f streamlit.log
```

---

## Step 9: Access Your Application

Open your web browser and navigate to:
`http://<your-ec2-public-ip>/askbot`

You should now see the SourceIQ RAG Engine interface! From here, you can upload documents, input your Google API Key, and start chatting.
