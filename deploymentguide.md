# Deployment Guide for LLM Security Benchmark Application

## 1. Application Overview

This Streamlit application integrates with Purple Llama's CyberSecEval 3 tool to benchmark the security of various Large Language Models (LLMs). The application features:

- A user interface for entering and managing API keys
- Local storage of API keys for convenience
- Unique result file naming based on model name and timestamp
- Visual indicators for successful API connections
- A results comparison page for up to 4 models
- Integration with GitHub via Model Context Protocol (MCP)

## 2. Environment Setup with Windsurf IDE

### System Requirements
- macOS (tested on M3 Max with 128GB RAM)
- Python 3.9 or newer
- Git
- Node.js and npm
- Windsurf IDE by Codeium

### Installing Windsurf IDE

1. Download Windsurf IDE from the Codeium website (https://codeium.com/windsurf)
2. Install the application by following the on-screen instructions
3. Launch Windsurf and complete the initial setup:
   - Sign in with your Codeium account (create one if needed)
   - If you're migrating from VS Code, you can import your settings
   - Select your preferred keybindings and theme

### Installing Node.js and npm

Before setting up the GitHub MCP server, you need to install Node.js and npm:

1. **Install Homebrew** (if you don't have it already):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. **Install Node.js and npm** using Homebrew:
   ```bash
   brew install node
   ```

3. **Verify the installation**:
   ```bash
   node --version
   npm --version
   ```

Alternative installation methods:
- You can download the installer directly from the [Node.js website](https://nodejs.org/)
- You can use version managers like nvm if you need to work with multiple Node.js versions

### Setting up the GitHub MCP Server

The GitHub MCP server installation is done through your Mac's terminal, not directly within Windsurf:

1. **Choose an installation location**:
   - You can install the GitHub MCP server in any directory on your Mac
   - Create a dedicated directory for MCP servers:

   ```bash
   # Create and navigate to your chosen directory
   mkdir -p ~/mcp-servers
   cd ~/mcp-servers
   ```

2. **Clone and set up the MCP server**:
   ```bash
   # Clone the MCP servers repository
   git clone https://github.com/modelcontextprotocol/servers.git github-mcp
   cd github-mcp

   # Install dependencies
   npm install

   # Build the GitHub server
   npm run build
   ```

3. **Remember the installation path**:
   - Make note of the full path where you installed the server
   - You can get the full path by running: `pwd` in that directory
   - You'll need this path when configuring Windsurf

4. **Generate a GitHub Personal Access Token**:

   #### Understanding the Required Scopes

   - **`repo` scope**: Grants full control of repositories (code, issues, PRs, settings)
   - **`workflow` scope**: Allows management of GitHub Actions workflows
   - **`user` scope**: Provides access to user profile information

   #### Step-by-Step Token Generation

   1. **Navigate to GitHub Settings**:
      - Log in to your GitHub account
      - Click on your profile photo in the top-right corner
      - Select "Settings" from the dropdown menu

   2. **Access Developer Settings**:
      - Scroll to the bottom of the left sidebar
      - Click on "Developer settings"

   3. **Create a Personal Access Token**:
      - In the left sidebar, click "Personal access tokens"
      - Select "Tokens (classic)"
      - Click "Generate new token"
      - Choose "Generate new token (classic)"

   4. **Configure the Token**:
      - Give your token a descriptive name (e.g., "GitHub MCP Server")
      - Set an expiration date (choose based on your security requirements)
      - Select the following scopes by checking their boxes:
        - Under "repo", check the entire section to grant full repository access
        - Under "workflow", check the box to grant GitHub Actions workflow control
        - Under "user", check the box for user information access

   5. **Generate and Copy the Token**:
      - Scroll to the bottom and click "Generate token"
      - **IMPORTANT**: Copy the generated token immediately and store it securely
      - You won't be able to see this token again after navigating away from the page

5. **Configure the GitHub MCP server**:
   ```bash
   # Create a config directory for MCP
   mkdir -p ~/.config/mcp
   touch ~/.config/mcp/github.json
   ```

6. **Edit `~/.config/mcp/github.json` with the following content**:
   ```json
   {
     "github": {
       "token": "YOUR_COPIED_TOKEN"
     }
   }
   ```

7. **Configure Windsurf to use the GitHub MCP server**:
   - Open Windsurf settings (click the gear icon or use ⌘+, on Mac)
   - Navigate to the MCP Servers section
   - Add a new MCP server with the following configuration:

   ```json
   {
     "mcpServers": {
       "github": {
         "command": "npx",
         "args": ["-y", "@modelcontextprotocol/server-github"],
         "env": {
           "GITHUB_PERSONAL_ACCESS_TOKEN": "YOUR_COPIED_TOKEN"
         }
       }
     }
   }
   ```

   **Security Best Practices for GitHub Tokens**:
   - Never share your token publicly
   - Consider setting an expiration date for your token
   - Use the minimum necessary permissions for your use case
   - Store the token securely (e.g., in a password manager)
   - Rotate tokens periodically for enhanced security

8. **Start the GitHub MCP server in Windsurf**:
   - Open the Cascade panel in Windsurf (usually on the right side)
   - Click on the MCP Servers tab
   - Refresh the servers list
   - Verify that the GitHub server is connected

## 3. Setting Up a New Project in Windsurf

### Creating a GitHub Repository

1. In Windsurf, open the Cascade panel
2. Use the GitHub MCP server to create a new repository:
   - Type: "Create a new GitHub repository named 'llm-security-benchmark'"
   - Specify your GitHub username when prompted
   - Choose whether the repository should be public or private

3. Clone the repository locally:
   - In Windsurf, use the built-in terminal (View > Terminal)
   - Run: `git clone https://github.com/yourusername/llm-security-benchmark.git`
   - Open the cloned repository in Windsurf (File > Open Folder)

### Creating Basic Project Structure

In Windsurf, use Cascade to create the project structure by typing:

"Create the following project structure for our LLM security benchmark application:
- app/ (directory for our Streamlit application)
- data/ (directory for benchmark results)
- config/ (directory for configuration files)
- logs/ (directory for log files)
- README.md (project overview)
- CHANGELOG.md (track changes)
- RECOMMENDATIONS.md (suggestions for improvements)
- .gitignore (ignore sensitive files)"