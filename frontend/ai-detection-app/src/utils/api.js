import axios from 'axios';

const API_BASE_URL = '/api';

// Flag to enable demo mode when backend is not available
const DEMO_MODE = true;

// Create axios instance with default config
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Generate mock response for demo mode
const generateMockResponse = (type, fileType) => {
  // For deepfake detection, stronger bias toward authenticity (85% chance of being authentic)
  const isPositive = type === 'deepfake' 
    ? Math.random() > 0.85 
    : Math.random() > 0.6;
  
  // Higher confidence ranges for more realistic model behavior
  const confidence = isPositive ? 
    Math.random() * 0.2 + 0.8 : // 0.8 to 1.0 for positive detection
    Math.random() * 0.15 + 0.8; // 0.8 to 0.95 for negative detection (high confidence in authenticity)
  
  let result = {
    timestamp: new Date().toISOString()
  };
  
  switch (type) {
    case 'deepfake':
      const rawScore = isPositive ? 
        Math.random() * 0.2 + 0.8 : // 0.8 to 1.0 for deepfakes
        Math.random() * 0.3;        // 0.0 to 0.3 for authentic videos
      
      const frameCount = Math.floor(Math.random() * 15) + 5; // 5-20 frames
      const frameStd = isPositive ? 
        Math.random() * 0.15 + 0.1 : // Higher variation for deepfakes
        Math.random() * 0.08;        // Lower variation for authentic
      
      const temporalConsistency = isPositive ?
        Math.random() * 0.4 + 0.2 :  // 0.2-0.6 for deepfakes (less consistent)
        Math.random() * 0.25 + 0.7;  // 0.7-0.95 for authentic (more consistent)
      
      result = {
        ...result,
        is_deepfake: isPositive,
        confidence: confidence,
        raw_score: rawScore,
        frames_analyzed: frameCount,
        frame_std: frameStd,
        temporal_consistency: temporalConsistency,
        source: 'demo_video.mp4'
      };
      break;
      
    case 'document':
      const forgeryTypes = ["content_manipulation", "signature_forgery", "digital_splicing", "ai_generated"];
      const typeIndex = Math.floor(Math.random() * forgeryTypes.length);
      const forgeryType = forgeryTypes[typeIndex];
      const rawDocScore = isPositive ? 
        Math.random() * 0.3 + 0.7 : // 0.7 to 1.0 for forged
        Math.random() * 0.4;        // 0.0 to 0.4 for authentic
      
      // Create additional analysis results
      const additionalAnalysis = {
        metadata_tampering: isPositive && Math.random() > 0.5,
        text_consistency: isPositive && (forgeryType === "content_manipulation" || forgeryType === "ai_generated"),
        visual_artifacts: isPositive && forgeryType === "digital_splicing",
        color_inconsistencies: isPositive && (forgeryType === "content_manipulation" || Math.random() > 0.7)
      };
      
      result = {
        ...result,
        is_forged: isPositive,
        forgery_confidence: confidence,
        forgery_type: forgeryType,
        type_confidence: confidence * 0.9,
        raw_score: rawDocScore,
        additional_analysis: additionalAnalysis,
        source: 'demo_document.' + fileType
      };
      break;
      
    case 'signature':
      // For signatures, calculate similarity score (higher means more similar)
      const similarityScore = isPositive ? 
        Math.random() * 0.3 :         // 0.0 to 0.3 for forged (low similarity)
        Math.random() * 0.4 + 0.6;    // 0.6 to 1.0 for authentic (high similarity)
      
      // Create signature analysis
      const signatureAnalysis = {
        pen_pressure_consistent: similarityScore > 0.5,
        stroke_patterns_match: similarityScore > 0.7,
        proportions_consistent: similarityScore > 0.6,
        characteristic_features_match: similarityScore > 0.65
      };
      
      result = {
        ...result,
        is_forged: isPositive,
        confidence: confidence,
        similarity_score: similarityScore,
        signature_analysis: signatureAnalysis,
        reference: 'reference_signature.' + fileType,
        query: 'query_signature.' + fileType
      };
      break;
      
    default:
      break;
  }
  
  return result;
};

// API endpoints
export const detectDeepfake = async (videoFile) => {
  if (DEMO_MODE) {
    // Simulate network delay
    await new Promise(resolve => setTimeout(resolve, 2000));
    return generateMockResponse('deepfake', videoFile.name.split('.').pop());
  }
  
  const formData = new FormData();
  formData.append('file', videoFile);
  
  try {
    const response = await apiClient.post('/detect/deepfake', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  } catch (error) {
    console.error('Error detecting deepfake:', error);
    throw error;
  }
};

export const detectDocumentForgery = async (documentFile) => {
  if (DEMO_MODE) {
    // Simulate network delay
    await new Promise(resolve => setTimeout(resolve, 1500));
    return generateMockResponse('document', documentFile.name.split('.').pop());
  }
  
  const formData = new FormData();
  formData.append('file', documentFile);
  
  try {
    const response = await apiClient.post('/detect/document', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  } catch (error) {
    console.error('Error detecting document forgery:', error);
    throw error;
  }
};

export const detectSignatureForgery = async (signatureFile, referenceFile) => {
  if (DEMO_MODE) {
    // Simulate network delay
    await new Promise(resolve => setTimeout(resolve, 1500));
    return generateMockResponse('signature', signatureFile.name.split('.').pop());
  }
  
  const formData = new FormData();
  formData.append('file', signatureFile);
  
  if (referenceFile) {
    formData.append('reference', referenceFile);
  }
  
  try {
    const response = await apiClient.post('/detect/signature', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  } catch (error) {
    console.error('Error detecting signature forgery:', error);
    throw error;
  }
};

export const submitReport = async (reportData) => {
  if (DEMO_MODE) {
    // Simulate network delay
    await new Promise(resolve => setTimeout(resolve, 500));
    return { 
      success: true, 
      id: Math.random().toString(36).substring(2, 10),
      message: 'Report submitted successfully'
    };
  }
  
  try {
    const response = await apiClient.post('/report', reportData);
    return response.data;
  } catch (error) {
    console.error('Error submitting report:', error);
    throw error;
  }
};

// Get detection statistics
export const getStatistics = async () => {
  if (DEMO_MODE) {
    // Simulate network delay
    await new Promise(resolve => setTimeout(resolve, 700));
    return {
      total_detections: Math.floor(Math.random() * 5000) + 1000,
      deepfake_detections: Math.floor(Math.random() * 2000) + 500,
      document_forgery_detections: Math.floor(Math.random() * 1500) + 300,
      signature_forgery_detections: Math.floor(Math.random() * 1000) + 200,
      detection_accuracy: Math.floor(Math.random() * 10) + 90 // 90-99%
    };
  }
  
  try {
    const response = await apiClient.get('/statistics');
    return response.data;
  } catch (error) {
    console.error('Error fetching statistics:', error);
    throw error;
  }
};

export default {
  detectDeepfake,
  detectDocumentForgery,
  detectSignatureForgery,
  submitReport,
  getStatistics,
}; 