import { useState, useRef } from 'react';
import { Form, Button, Alert, Spinner } from 'react-bootstrap';
import { detectDeepfake, detectDocumentForgery, detectSignatureForgery } from '../utils/api';

function DetectionForm({ detectionType, acceptedFileTypes, onSubmit }) {
  const [file, setFile] = useState(null);
  const [referenceFile, setReferenceFile] = useState(null); // For signature comparison
  const [isUploading, setIsUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState('');
  const [isDragging, setIsDragging] = useState(false);
  const [preview, setPreview] = useState(null);
  const [referencePreview, setReferencePreview] = useState(null);
  
  const fileInputRef = useRef(null);
  const referenceFileInputRef = useRef(null);
  
  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    validateAndSetFile(selectedFile, 'main');
  };
  
  const handleReferenceFileChange = (e) => {
    const selectedFile = e.target.files[0];
    validateAndSetFile(selectedFile, 'reference');
  };
  
  const validateAndSetFile = (selectedFile, fileType) => {
    setError('');
    
    if (!selectedFile) {
      if (fileType === 'main') {
        setFile(null);
        setPreview(null);
      } else {
        setReferenceFile(null);
        setReferencePreview(null);
      }
      return;
    }
    
    // Validate file type
    if (!acceptedFileTypes.includes(selectedFile.type)) {
      setError(`Invalid file type. Accepted types: ${acceptedFileTypes.join(', ')}`);
      if (fileType === 'main') {
        setFile(null);
        setPreview(null);
      } else {
        setReferenceFile(null);
        setReferencePreview(null);
      }
      return;
    }
    
    // Set file
    if (fileType === 'main') {
      setFile(selectedFile);
    } else {
      setReferenceFile(selectedFile);
    }
    
    // Create preview for image files
    if (selectedFile.type.startsWith('image/')) {
      const reader = new FileReader();
      reader.onloadend = () => {
        if (fileType === 'main') {
          setPreview(reader.result);
        } else {
          setReferencePreview(reader.result);
        }
      };
      reader.readAsDataURL(selectedFile);
    } else if (selectedFile.type.startsWith('video/')) {
      // For video, we'll just show a placeholder icon
      if (fileType === 'main') {
        setPreview('video');
      } else {
        setReferencePreview('video');
      }
    } else if (selectedFile.type === 'application/pdf') {
      // For PDF, we'll just show a placeholder icon
      if (fileType === 'main') {
        setPreview('pdf');
      } else {
        setReferencePreview('pdf');
      }
    } else {
      if (fileType === 'main') {
        setPreview(null);
      } else {
        setReferencePreview(null);
      }
    }
  };
  
  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };
  
  const handleDragLeave = () => {
    setIsDragging(false);
  };
  
  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    
    const droppedFile = e.dataTransfer.files[0];
    validateAndSetFile(droppedFile, 'main');
  };
  
  const handleUploadClick = () => {
    fileInputRef.current.click();
  };
  
  const handleReferenceUploadClick = () => {
    referenceFileInputRef.current.click();
  };
  
  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!file) {
      setError('Please select a file to analyze');
      return;
    }
    
    if (detectionType === 'Signature Forgery' && !referenceFile) {
      setError('Please select a reference signature to compare with');
      return;
    }
    
    setError('');
    setIsUploading(true);
    setProgress(0);
    
    // Simulate progress
    const progressInterval = setInterval(() => {
      setProgress(prevProgress => {
        const nextProgress = prevProgress + 10;
        return nextProgress > 90 ? 90 : nextProgress;
      });
    }, 300);
    
    try {
      let apiResult;
      
      // Call appropriate API based on detection type
      switch (detectionType) {
        case 'Video Deepfake':
          apiResult = await detectDeepfake(file);
          break;
        case 'Document Forgery':
          apiResult = await detectDocumentForgery(file);
          break;
        case 'Signature Forgery':
          apiResult = await detectSignatureForgery(file, referenceFile);
          break;
        default:
          throw new Error('Invalid detection type');
      }
      
      clearInterval(progressInterval);
      setProgress(100);
      
      // Format the result for the ResultPanel component
      let resultData = {
        detectionType,
        filename: file.name,
        timestamp: apiResult.timestamp || new Date().toISOString(),
        thumbnailUrl: preview !== 'video' && preview !== 'pdf' ? preview : null,
        detectionFeatures: []
      };
      
      // Process results based on detection type
      if (detectionType === 'Video Deepfake') {
        resultData.isAI = apiResult.is_deepfake;
        resultData.confidenceScore = Math.round(apiResult.confidence * 100);
        
        // Create basic detection features
        const detectionFeatures = [
          {
            name: 'Detection Result',
            detected: apiResult.is_deepfake,
            value: apiResult.is_deepfake ? 'Deepfake Detected' : 'Authentic Video'
          },
          {
            name: 'Confidence Level',
            value: `${Math.round(apiResult.confidence * 100)}%`
          }
        ];
        
        // Add frame analysis info
        if (apiResult.frames_analyzed) {
          detectionFeatures.push({
            name: 'Analysis Depth',
            value: `Analyzed ${apiResult.frames_analyzed} frames`
          });
        }
        
        // Add temporal consistency if available
        if (apiResult.temporal_consistency !== undefined) {
          detectionFeatures.push({
            name: 'Temporal Consistency',
            detected: apiResult.temporal_consistency < 0.7,
            value: apiResult.temporal_consistency > 0.7 
              ? 'High (natural motion)' 
              : 'Low (possible manipulation)'
          });
        }
        
        // Add additional technical details for debugging
        if (apiResult.raw_score !== undefined) {
          detectionFeatures.push({
            name: 'Technical Details',
            value: `Raw score: ${apiResult.raw_score.toFixed(3)}${apiResult.frame_std ? `, Frame variation: ${apiResult.frame_std.toFixed(3)}` : ''}`
          });
        }
        
        resultData.detectionFeatures = detectionFeatures;
        
      } else if (detectionType === 'Document Forgery') {
        resultData.isAI = apiResult.is_forged;
        resultData.confidenceScore = Math.round(apiResult.forgery_confidence * 100);
        resultData.forgeryType = apiResult.forgery_type;
        
        // Create basic detection features
        const detectionFeatures = [
          {
            name: 'Analysis Result',
            detected: apiResult.is_forged,
            value: apiResult.is_forged ? 'Forged Document' : 'Authentic Document'
          },
          {
            name: 'Confidence',
            value: `${Math.round(apiResult.forgery_confidence * 100)}%`
          }
        ];
        
        // Add forgery type if document is forged
        if (apiResult.is_forged) {
          const forgeryTypeMap = {
            'content_manipulation': 'Content Manipulation',
            'signature_forgery': 'Signature Forgery',
            'digital_splicing': 'Digital Splicing',
            'ai_generated': 'AI Generated'
          };
          
          detectionFeatures.push({
            name: 'Forgery Type',
            value: forgeryTypeMap[apiResult.forgery_type] || apiResult.forgery_type,
            confidence: Math.round(apiResult.type_confidence * 100)
          });
        }
        
        // Add detailed analysis if available
        if (apiResult.additional_analysis) {
          const analysis = apiResult.additional_analysis;
          
          if (analysis.text_consistency !== undefined) {
            detectionFeatures.push({
              name: 'Text Consistency',
              detected: analysis.text_consistency,
              value: analysis.text_consistency ? 'Inconsistencies detected' : 'Consistent'
            });
          }
          
          if (analysis.visual_artifacts !== undefined) {
            detectionFeatures.push({
              name: 'Visual Artifacts',
              detected: analysis.visual_artifacts,
              value: analysis.visual_artifacts ? 'Present' : 'Not detected'
            });
          }
          
          if (analysis.color_inconsistencies !== undefined) {
            detectionFeatures.push({
              name: 'Color Analysis',
              detected: analysis.color_inconsistencies,
              value: analysis.color_inconsistencies ? 'Inconsistent' : 'Consistent'
            });
          }
        }
        
        // Add raw score for technical users
        if (apiResult.raw_score !== undefined) {
          detectionFeatures.push({
            name: 'Technical Info',
            value: `Raw score: ${apiResult.raw_score.toFixed(3)}`
          });
        }
        
        resultData.detectionFeatures = detectionFeatures;
        
      } else if (detectionType === 'Signature Forgery') {
        resultData.isAI = apiResult.is_forged;
        resultData.confidenceScore = Math.round(apiResult.confidence * 100);
        
        // Create basic detection features
        const detectionFeatures = [
          {
            name: 'Authentication Result',
            detected: !apiResult.is_forged,
            value: apiResult.is_forged ? 'Signatures do not match' : 'Signatures match'
          },
          {
            name: 'Confidence',
            value: `${Math.round(apiResult.confidence * 100)}%`
          }
        ];
        
        // Add similarity score if available
        if (apiResult.similarity_score !== undefined) {
          detectionFeatures.push({
            name: 'Similarity Score',
            value: `${Math.round(apiResult.similarity_score * 100)}%`
          });
        }
        
        // Add detailed signature analysis if available
        if (apiResult.signature_analysis) {
          const analysis = apiResult.signature_analysis;
          
          if (analysis.stroke_patterns_match !== undefined) {
            detectionFeatures.push({
              name: 'Stroke Patterns',
              detected: analysis.stroke_patterns_match,
              value: analysis.stroke_patterns_match ? 'Match' : 'Different'
            });
          }
          
          if (analysis.pen_pressure_consistent !== undefined) {
            detectionFeatures.push({
              name: 'Pen Pressure',
              detected: analysis.pen_pressure_consistent,
              value: analysis.pen_pressure_consistent ? 'Consistent' : 'Inconsistent'
            });
          }
          
          if (analysis.proportions_consistent !== undefined) {
            detectionFeatures.push({
              name: 'Proportions',
              detected: analysis.proportions_consistent,
              value: analysis.proportions_consistent ? 'Consistent' : 'Inconsistent'
            });
          }
        }
        
        resultData.detectionFeatures = detectionFeatures;
        resultData.referenceImage = referencePreview;
      }
      
      // Call the parent component's onSubmit function with the result
      onSubmit(resultData);
      
      // Reset form after successful upload
      setTimeout(() => {
        setFile(null);
        setReferenceFile(null);
        setPreview(null);
        setReferencePreview(null);
        setIsUploading(false);
        setProgress(0);
      }, 1000);
      
    } catch (err) {
      clearInterval(progressInterval);
      console.error('Error during detection:', err);
      setError('An error occurred during analysis. Please try again.');
      setIsUploading(false);
      setProgress(0);
    }
  };
  
  const renderFilePreview = (previewType, fileObj, isReference = false) => {
    if (!previewType) return null;
    
    if (previewType === 'video') {
      return (
        <div className="preview-icon video">
          <i className="fas fa-video fa-3x"></i>
          <span className="mt-2">{fileObj?.name}</span>
        </div>
      );
    } else if (previewType === 'pdf') {
      return (
        <div className="preview-icon pdf">
          <i className="fas fa-file-pdf fa-3x"></i>
          <span className="mt-2">{fileObj?.name}</span>
        </div>
      );
    } else {
      return (
        <div className="preview-image">
          <img src={previewType} alt={isReference ? "Reference" : "Preview"} className="img-fluid" />
          <span className="mt-1">{fileObj?.name}</span>
        </div>
      );
    }
  };
  
  return (
    <Form onSubmit={handleSubmit}>
      {error && (
        <Alert variant="danger" className="animate__animated animate__fadeIn">
          <i className="fas fa-exclamation-circle me-2"></i>
          {error}
        </Alert>
      )}
      
      <div 
        className={`upload-area ${isDragging ? 'drag-active' : ''} ${file ? 'has-file' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={handleUploadClick}
      >
        <input
          type="file"
          ref={fileInputRef}
          className="d-none"
          accept={acceptedFileTypes.join(',')}
          onChange={handleFileChange}
        />
        
        {file ? (
          renderFilePreview(preview, file)
        ) : (
          <div className="upload-prompt text-center">
            <div className="upload-icon">
              <i className="fas fa-upload fa-3x"></i>
            </div>
            <h5 className="mt-3 mb-2">Drag & Drop or Click to Upload</h5>
            <p className="text-muted mb-0">
              {detectionType === 'Video Deepfake' && 'Upload a video file to analyze for deepfakes'}
              {detectionType === 'Document Forgery' && 'Upload a document to check for forgery'}
              {detectionType === 'Signature Forgery' && 'Upload a signature image to verify authenticity'}
            </p>
            <p className="text-muted mt-2 small">
              Accepted formats: {acceptedFileTypes.map(type => type.split('/')[1]).join(', ')}
            </p>
          </div>
        )}
      </div>
      
      {/* Reference signature upload area (only for signature detection) */}
      {detectionType === 'Signature Forgery' && (
        <div className="mt-3">
          <p className="mb-2">Reference Signature:</p>
          <div 
            className={`upload-area upload-area-secondary ${referenceFile ? 'has-file' : ''}`}
            onClick={handleReferenceUploadClick}
          >
            <input
              type="file"
              ref={referenceFileInputRef}
              className="d-none"
              accept={acceptedFileTypes.join(',')}
              onChange={handleReferenceFileChange}
            />
            
            {referenceFile ? (
              renderFilePreview(referencePreview, referenceFile, true)
            ) : (
              <div className="upload-prompt text-center">
                <div className="upload-icon">
                  <i className="fas fa-signature fa-2x"></i>
                </div>
                <h6 className="mt-2 mb-1">Upload Reference Signature</h6>
                <p className="text-muted small mb-0">
                  Upload an authentic signature for comparison
                </p>
              </div>
            )}
          </div>
        </div>
      )}
      
      {file && !isUploading && (
        <div className="text-center mt-3 animate__animated animate__fadeIn">
          <Button 
            variant="outline-secondary" 
            size="sm" 
            className="rounded-pill me-2" 
            onClick={() => {
              setFile(null);
              setPreview(null);
            }}
          >
            <i className="fas fa-times me-1"></i> Remove
          </Button>
          <Button 
            type="submit" 
            variant="primary" 
            size="sm" 
            className="rounded-pill"
          >
            <i className="fas fa-search me-1"></i> Analyze
          </Button>
        </div>
      )}
      
      {isUploading && (
        <div className="mt-3 text-center animate__animated animate__fadeIn">
          <div className="progress mb-2">
            <div 
              className="progress-bar progress-bar-striped progress-bar-animated" 
              style={{ width: `${progress}%` }}
            ></div>
          </div>
          <p className="text-muted small mb-0">
            <Spinner animation="border" size="sm" className="me-2" />
            Analyzing... Please wait
          </p>
        </div>
      )}
      
      <style jsx="true">{`
        .upload-area {
          border: 2px dashed #d1d1d1;
          border-radius: 10px;
          padding: 2rem;
          margin-bottom: 1rem;
          text-align: center;
          cursor: pointer;
          transition: all 0.3s ease;
          background-color: #f8f9fa;
          position: relative;
        }
        
        .upload-area:hover {
          border-color: var(--primary-color);
          background-color: #f0f4f8;
        }
        
        .upload-area.drag-active {
          border-color: var(--primary-color);
          background-color: #e8f0fe;
          box-shadow: 0 0 15px rgba(var(--primary-rgb), 0.2);
          transform: scale(1.01);
        }
        
        .upload-area.has-file {
          background-color: #fff;
          border-style: solid;
          border-color: var(--primary-color);
        }
        
        .upload-icon {
          width: 80px;
          height: 80px;
          border-radius: 50%;
          background: rgba(var(--primary-rgb), 0.1);
          display: flex;
          align-items: center;
          justify-content: center;
          margin: 0 auto 1rem;
          color: var(--primary-color);
        }
        
        .preview-icon {
          display: flex;
          flex-direction: column;
          align-items: center;
          justify-content: center;
          color: var(--primary-color);
        }
        
        .preview-image {
          display: flex;
          flex-direction: column;
          align-items: center;
        }
        
        .preview-image img {
          max-height: 200px;
          border-radius: 6px;
          box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        }
        
        .processing-indicator {
          padding: 1rem;
          border-radius: 10px;
          background-color: #f8f9fa;
        }
      `}</style>
    </Form>
  );
}

export default DetectionForm; 