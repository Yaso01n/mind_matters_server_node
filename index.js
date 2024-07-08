const express = require('express');
const dotenv = require('dotenv');
const cors = require('cors');

dotenv.config(); // Load environment variables from .env file

const connectionModule = require('./DataBase/connection');

const UserRoute = require('./routes/userRouter');
// const DoctorRoute = require('./routes/doctorRouter');
const OrganizationRoute = require('./routes/organizationRouter');
// const AppointmentRoute = require('./routes/appointmentRouter');
// const RecordRoute = require('./routes/recordRouter');

const MindMatters_app = express();

// Set CORS options
const corsOptions = {
  origin: ['http://localhost:3000', 'https://your-flutter-app-domain.com'], // Adjust the allowed origins as needed
  optionsSuccessStatus: 200,
};

MindMatters_app.use(cors(corsOptions));

const PORT = process.env.PORT || 3000;
const HOST = process.env.HOST || 'http://localhost'; // Define HOST using an environment variable or a default value

connectionModule.connect((err) => {
  if (err) {
    console.error('Error connecting to database:', err);
    return;
  }

  console.log('DATABASE CONNECTED');

  MindMatters_app.listen(PORT, () => {
    console.log(`SERVER: ${HOST}:${PORT}`);
  });
});

MindMatters_app.use(express.json());
MindMatters_app.use('/', UserRoute);
// MindMatters_app.use('/', DoctorRoute);
MindMatters_app.use('/', OrganizationRoute);
// MindMatters_app.use('/', AppointmentRoute);
// MindMatters_app.use('/', RecordRoute);

// Catch-all handler for all other requests
MindMatters_app.use((req, res, next) => {
  res.status(404).send('404: NOT_FOUND');
});








// const express = require('express');
// const dotenv = require('dotenv');
// const cors = require('cors');

// dotenv.config(); // Load environment variables from .env file

// const connectionModule = require('./DataBase/connection');

// const UserRoute = require('./routes/userRouter');
// // const DoctorRoute = require('./routes/doctorRouter');
// const OrganizationRoute = require('./routes/organizationRouter');
// // const AppointmentRoute = require('./routes/appointmentRouter');
// // const RecordRoute = require('./routes/recordRouter');


// const MindMatters_app = express();
// MindMatters_app.use(cors());
// const PORT =  process.env.PORT || 3000;

// // //====================================================================
// // MindMatters_app.use(cors({                     
// //   origin: 'http://localhost:3000',
// // }));

// connectionModule.connect((err) => {
//   if (err){
//     console.error('Error connecting to database:', err);
//     return;
//   }

//   console.log('DATABASE CONNECTED');
  
//   MindMatters_app.listen(PORT, () => {
//     console.log(`SERVER: https://mindmattersservernode-git-main-yaso01ns-projects.vercel.app${PORT}`);
//   });
// });
// //====================================================================
// MindMatters_app.use(express.json());
// MindMatters_app.use('/', UserRoute);
// // MindMatters_app.use('/', DoctorRoute);
// MindMatters_app.use('/', OrganizationRoute);
// // MindMatters_app.use('/', AppointmentRoute);
// // MindMatters_app.use('/', RecordRoute);


