# 🚀 GitHub Setup Guide

## Step-by-Step Instructions to Upload Your Project to GitHub

### 1. Configure Git (First Time Setup)

```bash
# Set your name and email (replace with your actual details)
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

### 2. Create a New Repository on GitHub

1. **Go to GitHub:** Visit [https://github.com](https://github.com) and log in
2. **Create New Repository:**
   - Click the **"+"** icon in the top-right corner
   - Select **"New repository"**
3. **Repository Settings:**
   - **Repository name:** `cardiotox-prediction` (or your preferred name)
   - **Description:** "Machine learning pipeline for predicting cardiotoxicity in HER2+ breast cancer patients"
   - **Visibility:** Choose **Public** or **Private**
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)
4. **Click "Create repository"**

### 3. Connect Your Local Repository to GitHub

After creating the repository, GitHub will show you instructions. Use these commands:

```bash
# Navigate to your project directory
cd /Users/kardokhkakabra/Downloads/cardiotox_work_4

# Add the GitHub repository as a remote (replace YOUR_USERNAME with your actual GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/cardiotox-prediction.git

# Verify the remote was added
git remote -v
```

### 4. Push Your Code to GitHub

```bash
# Push to GitHub (you'll be prompted for your GitHub username and password/token)
git push -u origin main
```

**Note:** If you get an authentication error, you may need to use a Personal Access Token instead of your password:
1. Go to GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Generate new token with `repo` permissions
3. Use this token instead of your password when prompted

### 5. Update README.md with Your GitHub Username

After pushing, you should update the README.md file to replace placeholders:

1. Edit `README.md` and replace all instances of `YOUR_USERNAME` with your actual GitHub username
2. Commit and push the changes:

```bash
git add README.md
git commit -m "Update README with GitHub username"
git push
```

### 6. Verify Your Repository

Visit your repository at: `https://github.com/YOUR_USERNAME/cardiotox-prediction`

You should see:
- ✅ All your project files
- ✅ README.md displayed on the main page
- ✅ Nice badges and formatting
- ✅ Project structure visible

### 7. Optional: Add Topics and Details

On your GitHub repository page:
1. Click the **⚙️ Settings** gear icon next to "About"
2. Add **topics/tags:** `machine-learning`, `healthcare`, `cardiotoxicity`, `xgboost`, `python`, `breast-cancer`
3. Add **website:** `https://huggingface.co/spaces/kardokh/CTRCD`
4. Click **Save changes**

### 8. Optional: Enable GitHub Pages (for documentation)

If you want to create a project website:
1. Go to **Settings** → **Pages**
2. Select **main** branch as source
3. Select **/ (root)** as folder
4. Click **Save**
5. Your site will be available at: `https://YOUR_USERNAME.github.io/cardiotox-prediction/`

---

## 📝 Quick Reference Commands

### Check Status
```bash
git status
```

### Make Changes
```bash
# After editing files
git add .
git commit -m "Your descriptive commit message"
git push
```

### View History
```bash
git log --oneline
```

### Create a New Branch
```bash
git checkout -b feature-branch-name
```

---

## ⚠️ Important Notes

### Large Files
- GitHub has a **100MB file size limit**
- The `.gitignore` file already excludes:
  - Large data files (`dataset/*.zip`, `dataset/*.rar`)
  - Virtual environments (`.venv/`)
  - Python cache files (`__pycache__/`)

### If You Encounter Issues

**Problem: File too large**
```bash
# If you accidentally added a large file
git rm --cached path/to/large/file
git commit -m "Remove large file"
git push
```

**Problem: Authentication failed**
- Use a Personal Access Token instead of password
- Or set up SSH keys for easier authentication

**Problem: Merge conflicts**
- Usually occurs when working from multiple locations
- Pull before pushing: `git pull origin main`

---

## 🔄 Making Future Updates

After the initial setup, updating your repository is simple:

```bash
# Navigate to your project
cd /Users/kardokhkakabra/Downloads/cardiotox_work_4

# Check what changed
git status

# Add all changes
git add .

# Commit with a descriptive message
git commit -m "Add new feature X" # or "Fix bug Y" or "Update documentation"

# Push to GitHub
git push
```

---

## 🌟 Showcase Your Work

Once published, you can:

1. **Add to your GitHub profile README** (pin the repository)
2. **Share on LinkedIn** with the repository link
3. **Include in your CV/Resume** with the link
4. **Reference in job applications** to demonstrate skills
5. **Use in your portfolio** website

---

## 📧 Need Help?

- **Git Documentation:** https://git-scm.com/doc
- **GitHub Guides:** https://guides.github.com/
- **GitHub Support:** https://support.github.com/

---

## ✅ Checklist

- [ ] Configure git with your name and email
- [ ] Create new repository on GitHub
- [ ] Add remote origin
- [ ] Push code to GitHub
- [ ] Update README.md with your username
- [ ] Add topics/tags to repository
- [ ] Verify everything looks good
- [ ] Share your repository!

---

**Your repository is ready to be shared with the world! 🎉**
