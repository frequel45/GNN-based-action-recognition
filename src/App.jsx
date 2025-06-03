import React, { useRef, useState, useEffect } from 'react';
import Webcam from 'react-webcam';
import * as tf from '@tensorflow/tfjs';
import * as faceDetection from '@tensorflow-models/face-detection';
import * as faceLandmarksDetection from '@tensorflow-models/face-landmarks-detection';
import { 
  Box, 
  Button, 
  Container, 
  Typography, 
  Paper, 
  Grid,
  TextField,
  Alert
} from '@mui/material';
import { styled } from '@mui/material/styles';

const StyledPaper = styled(Paper)(({ theme }) => ({
  padding: theme.spacing(3),
  margin: theme.spacing(2, 0),
  backgroundColor: '#f5f5f5'
}));

const StyledWebcam = styled(Webcam)({
  width: '100%',
  borderRadius: '8px'
});

function App() {
  const webcamRef = useRef(null);
  const [detector, setDetector] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [childInfo, setChildInfo] = useState({
    name: '',
    age: '',
    lastSeen: '',
    description: ''
  });
  const [alert, setAlert] = useState(null);

  useEffect(() => {
    loadModels();
  }, []);

  const loadModels = async () => {
    try {
      await tf.ready();
      const model = await faceDetection.createDetector(
        faceDetection.SupportedModels.MediaPipeFaceDetector,
        { runtime: 'tfjs' }
      );
      setDetector(model);
    } catch (error) {
      console.error('Error loading models:', error);
      setAlert({
        severity: 'error',
        message: 'Failed to load detection models. Please refresh the page.'
      });
    }
  };

  const handleInputChange = (e) => {
    setChildInfo({
      ...childInfo,
      [e.target.name]: e.target.value
    });
  };

  const handleCapture = async () => {
    if (!detector) return;

    setIsProcessing(true);
    try {
      const imageSrc = webcamRef.current.getScreenshot();
      const img = new Image();
      img.src = imageSrc;
      
      await new Promise((resolve) => {
        img.onload = resolve;
      });

      const faces = await detector.estimateFaces(img);
      
      if (faces.length > 0) {
        setAlert({
          severity: 'success',
          message: `Detected ${faces.length} face(s). Processing and storing information...`
        });
        // Here you would typically send this data to your backend
      } else {
        setAlert({
          severity: 'warning',
          message: 'No faces detected. Please try again.'
        });
      }
    } catch (error) {
      console.error('Error during detection:', error);
      setAlert({
        severity: 'error',
        message: 'An error occurred during face detection.'
      });
    }
    setIsProcessing(false);
  };

  return (
    <Container maxWidth="lg">
      <Box sx={{ my: 4 }}>
        <Typography variant="h3" component="h1" gutterBottom align="center">
          Child Missing Detection System
        </Typography>

        {alert && (
          <Alert severity={alert.severity} sx={{ mb: 2 }}>
            {alert.message}
          </Alert>
        )}

        <Grid container spacing={3}>
          <Grid item xs={12} md={6}>
            <StyledPaper elevation={3}>
              <Typography variant="h5" gutterBottom>
                Child Information
              </Typography>
              <Box component="form" sx={{ '& > :not(style)': { m: 1 } }}>
                <TextField
                  fullWidth
                  label="Child's Name"
                  name="name"
                  value={childInfo.name}
                  onChange={handleInputChange}
                />
                <TextField
                  fullWidth
                  label="Age"
                  name="age"
                  type="number"
                  value={childInfo.age}
                  onChange={handleInputChange}
                />
                <TextField
                  fullWidth
                  label="Last Seen Location"
                  name="lastSeen"
                  value={childInfo.lastSeen}
                  onChange={handleInputChange}
                />
                <TextField
                  fullWidth
                  label="Description"
                  name="description"
                  multiline
                  rows={4}
                  value={childInfo.description}
                  onChange={handleInputChange}
                />
              </Box>
            </StyledPaper>
          </Grid>

          <Grid item xs={12} md={6}>
            <StyledPaper elevation={3}>
              <Typography variant="h5" gutterBottom>
                Face Detection
              </Typography>
              <Box sx={{ position: 'relative' }}>
                <StyledWebcam
                  ref={webcamRef}
                  screenshotFormat="image/jpeg"
                  mirrored={true}
                />
                <Button
                  variant="contained"
                  color="primary"
                  onClick={handleCapture}
                  disabled={isProcessing || !detector}
                  sx={{ mt: 2 }}
                  fullWidth
                >
                  {isProcessing ? 'Processing...' : 'Capture and Analyze'}
                </Button>
              </Box>
            </StyledPaper>
          </Grid>
        </Grid>
      </Box>
    </Container>
  );
}

export default App;