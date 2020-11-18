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

package org.phalanxdev.mi.utils;

import java.util.HashMap;
import java.util.Map;

/**
 * Default messages implementation (used in absence of anything else) - just has english
 *
 * @author Mark Hall (mhall{[at]}waikato{[dot]}ac{[dot]}nz)
 * @version 1: $
 */
public class DefaultMIMessages implements IMIMessages {

  protected Map<String, String> m_messages = new HashMap<>();

  public DefaultMIMessages() {
    m_messages.put("Evaluator.Error.EvaluatorNotInitialized", "This Evaluator object has not been initialized.");
    m_messages.put("Evaluator.Error.EvaluationWasNotPerformed", "Evaluation was not performed.");
    m_messages.put("Evaluator.Message.UnableToPerformPercentageSplit", "Unable to perform a percentage split evaluation because there are fewer than 10 training instances.");
    m_messages.put("Evaluator.Message.UnableToPerformCrossValidation", "Unable to perform a {0} fold cross-validation because there are fewer training instances ({0}) than folds.");
    m_messages.put("Evaluator.Message.PerformingPercentageSplit", "Performing a percentage split ({0}%) evaluation...");
    m_messages.put("Evaluator.Message.PerformingCrossValidation", "Performing {0}-fold cross-validation...");
    m_messages.put("Evaluator.Message.TrainingModelForFold", "Training model for fold {0}...");
    m_messages.put("Evaluator.Message.UnableToPerformSeparateTestSetEval", "Unable to perform separate test set evaluation because there are no test instances available.");
    m_messages.put("Evaluator.Error.FinalClassifierHasNotBeenTrainedYet", "Final classifier has not been trained yet!");
    m_messages.put("Evaluator.Error.IncrementalEvalOnlyOnTestOrTrainingData", "Incremental evaluation can only be performed on a separate test set or on the training data for incremental schemes (prequential evaluation).");
  }

  @Override
  public String getString(String key) {
    return null;
  }

  @Override
  public String getString(String key, String... parameters) {
    String match = m_messages.get(key);
    if (match == null) {
      return key;
    }

    if (parameters != null && parameters.length > 0) {
      // replace parameters in string
      for (int i = 0; i < parameters.length; i++) {
        String replacement = parameters[i];
        match = match.replace("{" + i + "}", replacement);
      }
    }

    return match;
  }

  @Override
  public String getString(String key, Object... parameters) {
    String[] strings = new String[parameters.length];

    for (int i = 0; i < strings.length; i++) {
      strings[i] = parameters[i] != null ? parameters[i].toString() : "";
    }

    return getString(key, strings);
  }
}
