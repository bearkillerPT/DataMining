import logo from './logo.svg';
import './App.css';
import { useState } from 'react';
function App() {
  const params = {
    "Age": Number,
    "Gender": { 'Female': 0, 'Male': 1, 'Other': 2 },
    "Hypertension": { 'No': 0, 'Yes': 1 },
    "Heart Disease": { 'No': 0, 'Yes': 1 },
    "Ever Married": { 'No': 0, 'Yes': 1 },
    "Work Type": { 'Govt_job': 0, 'Never_worked': 1, 'Private': 2, 'Self-employed': 3, 'children': 4 },
    "Residence Type": { 'Rural': 0, 'Urban': 1 },
    "Average Glucose Level": Number,
    "Bmi": Number,
    "Smoking Status": { 'Unknown': 0, 'formerly smoked': 1, 'never smoked': 2, 'smokes': 3 }
  }
  const [res, setRes] = useState(null)
  const [age, setAge] = useState(20)
  const [gender, setGender] = useState(0)
  const [hypertension, setHypertension] = useState(0)
  const [heart_disease, setHeart_disease] = useState(0)
  const [ever_married, setEver_married] = useState(0)
  const [work_type, setWork_type] = useState(0)
  const [residence_type, setResidence_type] = useState(20)
  const [avg_glucose_level, setAvg_glucose_level] = useState(20)
  const [bmi, setBmi] = useState(20)
  const [smoking_status, setSmoking_status] = useState(0)
  const paramState = {
    "Age": setAge,
    "Gender": setGender,
    "Hypertension": setHypertension,
    "Heart Disease": setHeart_disease,
    "Ever Married": setEver_married,
    "Work Type": setWork_type,
    "Residence Type": setResidence_type,
    "Average Glucose Level": setAvg_glucose_level,
    "Bmi": setBmi,
    "Smoking Status": setSmoking_status
  }
  return (
    <div className="App">
      <div className='appHeader'>
        Machine Learning Algorithm Models for stroke prediction
      </div>
      <div className='appHeader'>
        Input paramethers
      </div>
      <div className="paramsContainer">
        {Object.keys(params).map((param, idx) => {
          if (params[param] == Number)
            return (
              <div className='inputContainer' key={idx}>
                <p className='inputTitle'>{param}</p>
                <input defaultValue={20} className='inputNumber' type={'number'} onChange={res => paramState[param](res.target.value)} />
              </div>
            )
          else return (
            <div className='inputContainer' key={idx}>
              <p className='inputTitle'>{param}</p>
              <select className='inputSelect' onChange={res => paramState[param](params[param][res.target.value])}>{
                Object.keys(params[param]).map((param2, idx2) => {
                  return (
                    <option className='inputOption' key={idx2} value={param2}>{param2}</option>
                  )
                })
              }</select>
            </div>
          )
        })

        }
      </div>
      <div className='footer'>
        <div className='submitContainer' onClick={() => {
          let query = "https://edapi.bearkillerpt.xyz/?age=" + age + "&gender=" + gender + "&hypertension=" + hypertension + "&heart_disease=" + heart_disease + "&ever_married=" + ever_married + "&work_type=" + work_type + "&Residence_type=" + residence_type + "&avg_glucose_level=" + avg_glucose_level + "&bmi=" + bmi + "&smoking_status=" + smoking_status
          console.log(query)

          fetch(query).then(res => res.json()).then(res => { setRes(res); console.log(res) })
        }}>
          Submit!
        </div>
        <div className='resultsContainer'>
          {res &&
            Object.keys(res).map((ml_alg, idx) => {
              return (
                <div className='algCard' key={idx}>
                  <p className='algCardTitle'>{ml_alg}</p>
                  {
                    Object.keys(res[ml_alg]).map(
                      (sample_type, idx2) => {
                        return (
                          <div className='algCardContent' key={idx2}>
                            <p className='algCardP'>{sample_type}</p>
                            <p className='algCardP'>{res[ml_alg][sample_type] == 0 ? "No risk!" : "Risk of Stroke!"}</p>
                          {idx2 != Object.keys(res[ml_alg]).length - 1 &&
                            <div className='algTypeDivider'/>
                          }
                          </div>
                        );
                  })}</div>
              )
            })
          }
        </div>

      </div>
      <div className='notebookContainer'>
      <a href="https://ed.bearkillerpt.xyz/Data_Mining_Report.pdf" className='algCardTitle'>Resume Report available here!</a>
      <br/>
        <a href="https://ed.bearkillerpt.xyz/notebook" className='algCardTitle'>The jupyter notebook, also available here!</a>
        <iframe src='https://ed.bearkillerpt.xyz/notebook' className='notebookIframe'/>
      </div>
    </div>
  );
}

export default App;
