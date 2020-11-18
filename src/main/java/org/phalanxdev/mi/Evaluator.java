/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

package org.phalanxdev.mi;

import java.util.ArrayList;
import java.util.List;
import org.phalanxdev.mi.utils.IMIMessages;
import org.phalanxdev.mi.utils.IMILogAdapter;
import org.phalanxdev.mi.utils.IMIVariableAdaptor;
import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Attribute;
import weka.core.BatchPredictor;
import weka.core.Environment;
import weka.core.EnvironmentHandler;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.LogHandler;
import weka.core.OptionHandler;
import weka.core.Utils;

import java.io.File;
import java.net.URISyntaxException;
import java.util.Random;
import weka.gui.Logger;

/**
 * Class that handles evaluating a PMI model.
 *
 * @author Mark Hall (mhall{[at]}waikato{[dot]}ac{[dot]}nz)
 * @version $Revision: $
 */
public class Evaluator {

  /**
   * The evaluation mode being used
   */
  protected EvalMode m_evaluationMode = EvalMode.NONE;

  /**
   * Default of 10
   */
  protected int m_xValFolds = 10;

  /**
   * Default 2/3 1/3
   */
  protected int m_percentageSplit = 66;

  protected int m_randomSeed = 1;

  /**
   * True to compute area under the curve metrics (requires predictions to be retained in memory)
   */
  protected boolean m_computeAUC;

  /**
   * True if IR metrics are to be recorded in the output row (if class is nominal)
   */
  protected boolean m_outputIRMetrics;

  /**
   * Preserve order (rather than randomly shuffling) for % split and x-val
   */
  protected boolean m_preserveOrder;

  /**
   * The actual evaluation object
   */
  protected Evaluation m_eval;

  /**
   * Template for instantiating a new classifier
   */
  protected Classifier m_templateClassifier;

  /**
   * Holds the classifier built on all the training data
   */
  protected Classifier m_classifier;

  /**
   * Holds the training data
   */
  protected Instances m_trainingData;

  /**
   * True if the last evaluation was successfully performed. Evaluation will not be performed if
   * there are fewer training instances that x-val folds, or fewer than 10 instances for a
   * percentage split.
   */
  protected boolean m_evalWasPerformed;

  protected IMIMessages m_messages;

  /**
   * Construct a new Evaluator.
   *
   * @param mode the evaluation mode to use
   * @param randomSeed a random seed for shuffling/stratification
   * @param computeAUC true if area under the curve metrics are to be computed
   * @param outputIRMetrics true if information retrieval metrics are to be computed
   * @param messages the messages handler
   */
  public Evaluator(EvalMode mode, int randomSeed, boolean computeAUC, boolean outputIRMetrics,
      IMIMessages messages) {
    m_evaluationMode = mode;
    m_randomSeed = randomSeed;
    m_computeAUC = computeAUC;
    m_outputIRMetrics = outputIRMetrics;
    m_messages = messages;
  }

  /**
   * Initialize the evaluator
   *
   * @param trainingData the training data
   * @param untrainedClassifier the untrained classifier to use
   * @throws Exception if a problem occurs
   */
  public void initialize(Instances trainingData, Classifier untrainedClassifier) throws Exception {
    m_eval = new Evaluation(trainingData);
    m_trainingData = trainingData;
    m_templateClassifier = untrainedClassifier;
  }

  /**
   * Initialize the evaluator. No prior class probabilities are/can be computed because all that is
   * available at initialization time is the training metadata (i.e. attribute information).
   *
   * @param trainingHeader the training metadata (i.e. an instances object containing only attribute
   * information and no actual instances)
   * @param trainedModel the trained model to evaluate
   * @throws Exception if a problem occurs
   */
  public void initializeNoPriors(Instances trainingHeader, Classifier trainedModel)
      throws Exception {
    m_eval = new Evaluation(trainingHeader);
    m_eval.useNoPriors();
    m_classifier = trainedModel;
    m_templateClassifier = trainedModel;
    m_templateClassifier = copyClassifierTemplate(); // untrained template
    m_trainingData = trainingHeader;
  }

  /**
   * Returns metadata on the evaluation output row returned after evaluation, based on the input
   * data characteristics (class type) and user-specified settings (e.g. output of IR metrics etc.)
   *
   * @param trainingHeader the training data header; if null, then assumes that initialize() has
   * been called in order to provide the full training set
   * @param stratifiedBatches true if evaluation is being performed on stratified batches of data
   * @return evaluation row metadata in the form of a list of Attribute objects.
   * @throws Exception if no training data information is available.
   */
  public List<Attribute> getEvalRowMetadata(Instances trainingHeader,
      boolean stratifiedBatches) throws Exception {

    if (trainingHeader == null && m_trainingData == null) {
      throw new Exception("Unable to determine output metadata as no "
          + "training data information provided");
    }
    Instances trainingInfo = trainingHeader != null ? trainingHeader : m_trainingData;
    if (trainingInfo.classIndex() < 0) {
      throw new Exception("Unable to determine output metadata as attribute is not "
          + "set in training data information provided");
    }
    List<Attribute> metadata = new ArrayList<>();
    if (m_evaluationMode != EvalMode.NONE) {
      metadata.add(new Attribute("Scheme name", (List<String>) null));
      metadata.add(new Attribute("Scheme options", (List<String>) null));
      metadata.add(new Attribute("Evaluation mode", (List<String>) null));
      if (stratifiedBatches) {
        metadata.add(new Attribute("Stratification value", (List<String>) null));
      }

      // basic evaluation fields
      metadata.add(new Attribute("Unclassified instances"));
      if (trainingInfo.classAttribute().isNominal()) {
        metadata.add(new Attribute("Correctly classified instances"));
        metadata.add(new Attribute("Incorrectly classified instances"));
        metadata.add(new Attribute("Percent correct"));
        metadata.add(new Attribute("Percent incorrect"));
      }
      metadata.add(new Attribute("Mean absolute error"));
      metadata.add(new Attribute("Root mean squared error"));
      if (trainingInfo.classAttribute().isNumeric()) {
        metadata.add(new Attribute("Correlation coefficient"));
      }

      if (m_evaluationMode != EvalMode.PREQUENTIAL) {
        metadata.add(new Attribute("Relative absolute error"));
        metadata.add(new Attribute("Root relative squared error"));
      }

      metadata.add(new Attribute("Total number of instances"));
      if (trainingInfo.classAttribute().isNominal()) {
        metadata.add(new Attribute("Kappa statistic"));

        if (m_outputIRMetrics) {
          for (int i = 0; i < trainingInfo.classAttribute().numValues(); i++) {
            String label = trainingInfo.classAttribute().value(i) + "_";
            metadata.add(new Attribute(label + "TP rate"));
            metadata.add(new Attribute(label + "FP rate"));
            metadata.add(new Attribute(label + "Precision"));
            metadata.add(new Attribute(label + "Recall"));
            metadata.add(new Attribute(label + "F-measure"));
            metadata.add(new Attribute(label + "MCC"));
          }
        }
        if (m_computeAUC) {
          for (int i = 0; i < trainingInfo.classAttribute().numValues(); i++) {
            String label = trainingInfo.classAttribute().value(i) + "_";
            metadata.add(new Attribute(label + "ROC area"));
            metadata.add(new Attribute(label + "PRC area"));
          }
        }
        metadata.add(new Attribute("Confusion matrix"));
      }
    }

    return metadata;
  }

  /**
   * Set the percentage for training data in a percentage split evaluation
   *
   * @param percentageSplit the percentage for the training data
   */
  public void setPercentageSplit(int percentageSplit) {
    m_percentageSplit = percentageSplit;
  }

  /**
   * Get the percentage for training data in a percentage split evaluation
   *
   * @return the percentage for the training data
   */
  public int getPercentageSplit() {
    return m_percentageSplit;
  }

  /**
   * Set the number of folds to use for a cross-validation evaluation
   *
   * @param folds the number of folds to use
   */
  public void setXValFolds(int folds) {
    m_xValFolds = folds;
  }

  /**
   * Get the number of folds to use for a cross-validation evaluation
   *
   * @return the number of folds to use
   */
  public int getXValFolds() {
    return m_xValFolds;
  }

  /**
   * Set the random seed to use
   *
   * @param randomSeed the random seed value to use
   */
  public void setRandomSeed(int randomSeed) {
    m_randomSeed = randomSeed;
  }

  /**
   * Get the random seed to use
   *
   * @return the random seed value to use
   */
  public int getRandomSeed() {
    return m_randomSeed;
  }

  /**
   * Set an initialized Evaluation object to use. Useful for evaluating loaded serialized models
   * that have an Evaluation object configured with training data class priors.
   *
   * @param eval the Evaluation object to use.
   */
  public void setEvaluation(Evaluation eval) {
    m_eval = eval;
  }

  /**
   * Get the classifier template object
   *
   * @return the classifier template object
   */
  public Classifier getClassifierTemplate() {
    return m_templateClassifier;
  }

  /**
   * Return the training data
   *
   * @return the training data
   */
  public Instances getTrainingData() {
    return m_trainingData;
  }

  /**
   * Set an already trained classifier to evaluate.
   *
   * @param trainedClassifier the trained classifier to evaluate
   */
  public void setTrainedClassifier(Classifier trainedClassifier) {
    m_classifier = trainedClassifier;
  }

  /**
   * Returns true if the last evaluation was performed. Evaluation will not be performed if there
   * are fewer than 10 instances for a percentage split or fewer instances than folds for a
   * cross-validation.
   *
   * @return true if the last evaluation was performed.
   */
  public boolean wasEvaluationPerformed() {
    return m_evalWasPerformed;
  }

  public static void enableClassifierLoggingIfSupported(Classifier classifier, Object logAdapter) {
    if (classifier instanceof LogHandler && logAdapter != null && logAdapter instanceof Logger) {
      ((LogHandler) classifier).setLog((Logger) logAdapter);
    }
  }

  public static void configureWekaEnvironmentHandler(Object handler, IMIVariableAdaptor vars) {
    if (handler instanceof EnvironmentHandler) {
      Environment env = new Environment();

      if (vars != null) {
        for (String var : vars.listVariables()) {
          // if ( var.startsWith( "Internal." ) ) {
          String value = vars.getVariable(var);
          if (value.toLowerCase().startsWith("file:")) {
            value = value.replace(" ", "%20");
            try {
              File temp = new File(new java.net.URI(value));
              value = temp.toString();
            } catch (URISyntaxException e) {
              e.printStackTrace();
            }
          }
          env.addVariable(var, value);
          //}
        }
      }

      ((EnvironmentHandler) handler).setEnvironment(env);
    }
  }

  /**
   * Perform an evaluation.
   *
   * @param separateTestData optional separate test data (used in separate test set evaluation)
   * @param log the logging object to use
   * @param vars Kettle environment variables
   * @throws Exception if a problem occurs
   */
  public void performEvaluation(Instances separateTestData, IMILogAdapter log,
      IMIVariableAdaptor vars)
      throws Exception {
    if (m_trainingData == null) {
      throw new IllegalStateException(
          m_messages.getString("Evaluator.Error.EvaluatorNotInitialized"));
    }

    m_evalWasPerformed = true;
    Random r = new Random(m_randomSeed);
    // shuffle the training data
    if (!m_preserveOrder && (m_evaluationMode == EvalMode.CROSS_VALIDATION
        || m_evaluationMode == EvalMode.PERCENTAGE_SPLIT)) {
      m_trainingData.randomize(r);
    }
    if (m_evaluationMode == EvalMode.PERCENTAGE_SPLIT) {
      if (m_trainingData.numInstances() < 10) {
        log.logBasic(m_messages
            .getString("Evaluator.Message.UnableToPerformPercentageSplit"));
        m_evalWasPerformed = false;
        return;
      }
      log.logBasic(m_messages
          .getString("Evaluator.Message.PerformingPercentageSplit",
              m_percentageSplit));
      int trainSize = (int) Math.round(m_trainingData.numInstances() * m_percentageSplit / 100);
      int testSize = m_trainingData.numInstances() - trainSize;

      Instances train = new Instances(m_trainingData, 0, trainSize);
      Instances test = new Instances(m_trainingData, trainSize, testSize);
      Classifier classifierCopy = copyClassifierTemplate();
      enableClassifierLoggingIfSupported(classifierCopy, log);
      configureWekaEnvironmentHandler(classifierCopy, vars);
      classifierCopy.buildClassifier(train);
      if (m_computeAUC || (m_templateClassifier instanceof BatchPredictor
          && ((BatchPredictor) m_templateClassifier)
          .implementsMoreEfficientBatchPrediction())) {
        m_eval.evaluateModel(classifierCopy, test);
      } else {
        for (int i = 0; i < test.numInstances(); i++) {
          m_eval.evaluateModelOnce(classifierCopy, test.instance(i));
        }
      }
    } else if (m_evaluationMode == EvalMode.CROSS_VALIDATION) {
      if (m_trainingData.numInstances() < m_xValFolds) {
        log.logBasic(m_messages
            .getString("Evaluator.Message.UnableToPerformCrossValidation", m_xValFolds,
                m_trainingData.numInstances()));
        m_evalWasPerformed = false;
        return;
      }
      log.logBasic(m_messages
          .getString("Evaluator.Message.PerformingCrossValidation", m_xValFolds));
      if (!m_preserveOrder && m_trainingData.classAttribute().isNominal()) {
        m_trainingData.stratify(m_xValFolds);
      }
      for (int i = 0; i < m_xValFolds; i++) {
        log.logDetailed(m_messages
            .getString("Evaluator.Message.TrainingModelForFold", (i + 1)));
        Classifier foldClassifier = copyClassifierTemplate();
        enableClassifierLoggingIfSupported(foldClassifier, log);
        configureWekaEnvironmentHandler(foldClassifier, vars);
        Instances train = m_trainingData.trainCV(m_xValFolds, i, r);
        m_eval.setPriors(train);
        foldClassifier.buildClassifier(train);
        Instances test = m_trainingData.testCV(m_xValFolds, i);
        log.logDetailed(m_messages
            .getString("Evaluator.Message.TestingModelForFold", (i + 1)));

        if (m_templateClassifier instanceof BatchPredictor
            && ((BatchPredictor) m_templateClassifier)
            .implementsMoreEfficientBatchPrediction()) {
          Instances testCopy = new Instances(test);
          for (int j = 0; j < testCopy.numInstances(); j++) {
            testCopy.instance(j).setClassMissing();
          }
          double[][] preds = ((BatchPredictor) foldClassifier).distributionsForInstances(testCopy);
          for (int j = 0; j < test.numInstances(); j++) {
            if (m_computeAUC) {
              m_eval.evaluateModelOnceAndRecordPrediction(preds[j], test.instance(j));
            } else {
              m_eval.evaluateModelOnce(preds[j], test.instance(j));
            }
          }
        } else {
          for (int j = 0; j < test.numInstances(); j++) {
            if (m_computeAUC) {
              m_eval.evaluateModelOnceAndRecordPrediction(foldClassifier, test.instance(j));
            } else {
              m_eval.evaluateModelOnce(foldClassifier, test.instance(j));
            }
          }
        }
      }
    } else if (m_evaluationMode == EvalMode.SEPARATE_TEST_SET && separateTestData != null) {
      if (separateTestData.numInstances() == 0) {
        log.logBasic(m_messages
            .getString("Evaluator.Message.UnableToPerformSeparateTestSetEval"));
        m_evalWasPerformed = false;
        return;
      }
      if (m_classifier == null) {
        throw new IllegalStateException(m_messages
            .getString("Evaluator.Error.FinalClassifierHasNotBeenTrainedYet"));
      }
      enableClassifierLoggingIfSupported(m_classifier, log);
      configureWekaEnvironmentHandler(m_classifier, vars);
      // log.logBasic( "Performing separate test set evaluation..." );
      if (m_computeAUC || (m_templateClassifier instanceof BatchPredictor
          && ((BatchPredictor) m_templateClassifier)
          .implementsMoreEfficientBatchPrediction())) {
        m_eval.evaluateModel(m_classifier, separateTestData);
      } else {
        for (int i = 0; i < separateTestData.numInstances(); i++) {
          m_eval.evaluateModelOnce(m_classifier, separateTestData.instance(i));
        }
      }
    }
  }

  /**
   * Performs incremental evaluation. Only applicable to separate test set mode and
   * non-BatchPredictors
   *
   * @param testInstance the test instance to process
   * @param log the logging object
   * @throws Exception if a problem occurs
   */
  public void performEvaluationIncremental(Instance testInstance, IMILogAdapter log)
      throws Exception {
    if (m_evaluationMode != EvalMode.SEPARATE_TEST_SET
        && m_evaluationMode != EvalMode.PREQUENTIAL) {
      throw new IllegalStateException(
          "Incremental evaluation can only be performed on a separate test set or "
              + "on the training data for incremental schemes (prequential evaluation).");
    }

    if (m_templateClassifier instanceof BatchPredictor && ((BatchPredictor) m_templateClassifier)
        .implementsMoreEfficientBatchPrediction()) {
      throw new Exception(m_messages
          .getString("Evaluator.Error.IncrementalEvalOnlyOnTestOrTrainingData"));
    }

    if (m_computeAUC) {
      m_eval.evaluateModelOnceAndRecordPrediction(m_classifier, testInstance);
    } else {
      m_eval.evaluateModelOnce(m_classifier, testInstance);
    }
  }

  /**
   * Build a final model using all of the available training data.
   *
   * @param log the log to write to
   * @param vars Kettle environment variables
   * @return a final model
   * @throws Exception if a problem occurs
   */
  public Classifier buildFinalModel(IMILogAdapter log, IMIVariableAdaptor vars) throws Exception {

    if (m_trainingData == null) {
      throw new IllegalStateException(
          m_messages.getString("Evaluator.Error.EvaluatorNotInitialized"));
    }

    m_classifier = copyClassifierTemplate();
    if (log != null) {
      log.logBasic(m_messages.getString("BasePMIStep.Info.BuildingFinalModel",
          m_classifier.getClass().getCanonicalName() + " " + Utils
              .joinOptions(((OptionHandler) m_classifier).getOptions())));
      enableClassifierLoggingIfSupported(m_classifier, log);
    }
    configureWekaEnvironmentHandler(m_classifier, vars);

    m_classifier.buildClassifier(m_trainingData);

    return m_classifier;
  }

  /**
   * Return a row containing evaluation metrics
   *
   * @param stratificationValue optional stratification value
   * @param batchNumber the current batch number (if applicable)
   * @param log log to use
   * @return a row of evaluation metrics
   */
  public Object[] getEvalRow(String stratificationValue, int batchNumber,
      IMILogAdapter log) {
    if (m_trainingData == null) {
      throw new IllegalStateException(
          m_messages.getString("Evaluator.Error.EvaluatorNotInitialized"));
    }
    if (m_evaluationMode == EvalMode.NONE || m_eval.numInstances() == 0) {
      return null;
    } else {
      List<Object> outputRow = new ArrayList<>();
      int i = 0;
      String schemeName = m_templateClassifier.getClass().getCanonicalName();
      schemeName = schemeName.substring(schemeName.lastIndexOf(".") + 1);
      if (batchNumber > 0) {
        schemeName = "" + batchNumber + "_" + schemeName;
      }
      outputRow.add(schemeName);
      String schemeOptions = Utils.joinOptions(((OptionHandler) m_templateClassifier).getOptions());
      outputRow.add(schemeOptions);

      String evalMode = m_evaluationMode.toString().toLowerCase();
      if (m_evaluationMode == EvalMode.PERCENTAGE_SPLIT) {
        evalMode += " " + m_percentageSplit + "% seed " + m_randomSeed;
      } else if (m_evaluationMode == EvalMode.CROSS_VALIDATION) {
        evalMode += " folds " + m_xValFolds + " seed " + m_randomSeed;
      }
      outputRow.add(evalMode);

      if (!SchemeUtils.isEmpty(stratificationValue)) {
        outputRow.add(stratificationValue);
      }

      outputRow.add(m_eval.unclassified());

      if (m_trainingData.classAttribute().isNominal()) {
        outputRow.add(m_eval.correct());
        outputRow.add(m_eval.incorrect());
        outputRow.add(m_eval.pctCorrect());
        outputRow.add(m_eval.pctIncorrect());
      }
      outputRow.add(m_eval.meanAbsoluteError());
      outputRow.add(m_eval.rootMeanSquaredError());

      if (m_trainingData.classAttribute().isNumeric()) {
        try {
          outputRow.add(m_eval.correlationCoefficient());
        } catch (Exception e) {
          e.printStackTrace();
        }
      }

      if (m_evaluationMode != EvalMode.PREQUENTIAL) {
        try {
          outputRow.add(m_eval.relativeAbsoluteError());
        } catch (Exception e) {
          e.printStackTrace();
        }
        outputRow.add(m_eval.rootRelativeSquaredError());
      }

      outputRow.add(m_eval.numInstances());

      if (m_trainingData.classAttribute().isNominal()) {
        outputRow.add(m_eval.kappa());

        if (m_outputIRMetrics) {
          for (int j = 0; j < m_trainingData.classAttribute().numValues(); j++) {
            outputRow.add(m_eval.truePositiveRate(j));
            outputRow.add(m_eval.falsePositiveRate(j));
            outputRow.add(m_eval.precision(j));
            outputRow.add(m_eval.recall(j));
            outputRow.add(m_eval.fMeasure(j));
            outputRow.add(m_eval.matthewsCorrelationCoefficient(j));
          }
        }

        if (m_computeAUC) {
          for (int j = 0; j < m_trainingData.classAttribute().numValues(); j++) {
            outputRow.add(m_eval.areaUnderROC(j));
            outputRow.add(m_eval.areaUnderPRC(j));
          }
        }

        try {
          String matrix = m_eval.toMatrixString();
          outputRow.add(matrix);
          if (log != null) {
            log.logBasic(matrix);
          }
        } catch (Exception ex) {
          ex.printStackTrace();
        }
      }
      return outputRow.toArray(new Object[0]);
    }
  }

  /**
   * Get the underlying Weka Evaluation object
   *
   * @return the underlying Weka evaluation object
   */
  public Evaluation getEvaluation() {
    if (m_trainingData == null) {
      throw new IllegalStateException(
          m_messages.getString("Evaluator.Error.EvaluatorNotInitialized"));
    }

    if (!m_evalWasPerformed) {
      throw new IllegalStateException(
          m_messages.getString("Evaluator.Error.EvaluationWasNotPerformed"));
    }

    return m_eval;
  }

  /**
   * Create a copy of the classifier template
   *
   * @return a copy of the classifier template
   * @throws Exception if a problem occurs
   */
  protected Classifier copyClassifierTemplate() throws Exception {
    return (Classifier) Utils
        .forName(Classifier.class, m_templateClassifier.getClass().getCanonicalName(),
            ((OptionHandler) m_templateClassifier).getOptions());
  }

  /**
   * Enum for evaluation modes
   */
  public enum EvalMode {NONE, PERCENTAGE_SPLIT, CROSS_VALIDATION, SEPARATE_TEST_SET, PREQUENTIAL;}
}
