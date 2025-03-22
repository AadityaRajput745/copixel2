# AI Detection System Frontend

A modern React application for the AI-generated content detection system.

## Features

- Video deepfake detection
- Document forgery detection
- Signature forgery detection
- Result visualization
- Reporting capabilities

## Technology Stack

- React 18
- React Router v6
- Bootstrap 5 / React Bootstrap
- Vite build system
- Axios for API calls

## Project Structure

```
ai-detection-app/
├── public/            # Static assets
├── src/
│   ├── assets/        # Images, styles, and other assets
│   ├── components/    # Reusable UI components
│   ├── hooks/         # Custom React hooks
│   ├── pages/         # Page components
│   ├── utils/         # Utility functions
│   ├── App.jsx        # Main application component
│   └── main.jsx       # Entry point
├── index.html         # HTML template
├── package.json       # Dependencies and scripts
└── vite.config.js     # Vite configuration
```

## Getting Started

1. **Prerequisites**
   - Node.js 14+ and npm installed

2. **Installation**
   ```bash
   # Install dependencies
   npm install
   ```

3. **Development**
   ```bash
   # Start development server
   npm run dev
   ```

4. **Build for Production**
   ```bash
   # Create optimized production build
   npm run build
   ```

## Backend Integration

This frontend application connects to a Flask backend API for AI detection functionality. The API proxy is configured in `vite.config.js` to forward requests to the backend server running on port 5000.

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build locally
- `npm run lint` - Lint code with ESLint

## Contributing

Please follow the existing code style and component structure when adding new features or making changes. 