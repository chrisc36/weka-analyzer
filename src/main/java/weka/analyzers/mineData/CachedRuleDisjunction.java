package weka.analyzers.mineData;

import java.util.BitSet;

/**
 * The Disjunction of other cached rules.
 *
 * @param <T> the type of CachedRule contained in this RuleSet
 */
public class CachedRuleDisjunction<T extends CachedRule> extends CachedRuleSet<T> {

    /**
     * Construct a new empty RuleDisjunction that covers all Instances.
     *
     * @param numInstances number of instances this RuleSet should cover
     */
    public CachedRuleDisjunction(int numInstances) {
        super(numInstances, "OR");
    }

    @Override
    protected void addBitSet(BitSet other) {
        covered.or(other);
    }

}
