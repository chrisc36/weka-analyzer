package weka.analyzers.mineData;

import weka.core.Attribute;
import weka.core.Instances;

import java.util.BitSet;

/**
 * CachedRule that applies to a single column.
 */
public class BasicCachedRule implements CachedRule {

    /**
     * Constructs a Rules that returns true for every Instance.
     *
     * @param size number of instances in the dataset.
     * @return The empty Rule
     */
    public static BasicCachedRule emptyRule(int size) {
        BitSet all = new BitSet(size);
        all.flip(0, size);
        return new BasicCachedRule(all, "true");
    }

    /**
     * Constructs a Rules that covers instances if they have a particular value
     * in a specific attribute.
     *
     * @param data Instances the rule will apply to
     * @param att Attribute the rule applies to
     * @param val double to test equality for
     * @return CachedRule covering instances with val in att
     */
    public static BasicCachedRule EqualsRule(Instances data, Attribute att,
                                             double val) {
        BitSet covered = new BitSet(data.numInstances());
        for(int i = 0; i < data.numInstances(); i++) {
            if(data.instance(i).value(att) == val) {
                covered.set(i);
            }
        }
        return new BasicCachedRule(covered, val, "==", att);
    }

    /**
     * Constructs a rule that covers instances that do not have a given value
     * for a given attribute.
     *
     * @param data Instances the rule will apply to
     * @param att Attribute the rule applies to
     * @param val double to check attributes values are not equals to
     * @return CachedRule covering Instances without val in att
     */
    public static BasicCachedRule NotEqualsRule(Instances data, Attribute att,
                                                double val) {
        BitSet covered = new BitSet(data.numInstances());
        for(int i = 0; i < data.numInstances(); i++) {
            if(data.instance(i).value(att) != val) {
                covered.set(i);
            }
        }
        return new BasicCachedRule(covered, val, "!=", att);
    }

    /**
     * Constructs a CachedRule that covers an instance iff that instance has
     * a values greater then another value in a specific column
     *
     * @param data Instances the rule will apply to
     * @param att Attribute to build the rule from
     * @param val double the instance's value at {}att must be larger then
     * @return CachedRule covering Instances larger with values
     */
    public static BasicCachedRule GreaterThanRule(Instances data, Attribute att,
                                                  double val) {
        BitSet covered = new BitSet(data.numInstances());
        for(int i = 0; i < data.numInstances(); i++) {
            if(data.instance(i).value(att) > val) {
                covered.set(i);
            }
        }
        return new BasicCachedRule(covered, val, ">", att);
    }

    /**
     * Constructs a CachedRule that covers an instance iff that instance has
     * a values smaller then another value in a specific column
     *
     * @param data Instances the rule will apply to
     * @param att Attribute the rule applies to
     * @param val double the instance's value must be smaller then
     * @return CachedRule covering Instances with values smaller then val in att
     */
    public static BasicCachedRule SmallerThanRule(Instances data, Attribute att,
                                                  double val) {
        BitSet covered = new BitSet(data.numInstances());
        for(int i = 0; i < data.numInstances(); i++) {
            if(data.instance(i).value(att) < val) {
                covered.set(i);
            }
        }
        return new BasicCachedRule(covered, val, "<", att);
    }

    /**
     * Constructs a CachedRule that covers an instance iff that instance has
     * a values smaller or equal to another value in a specific column
     *
     * @param data Instances the rule will apply to
     * @param att Attribute to build the rule from
     * @param val double the instance's value must be smaller or equal to
     * @return CachedRule covering Instances with smaller or equal values
     */
    public static BasicCachedRule SmallerOrEqualRule(Instances data, Attribute att,
                                                     double val) {
        BitSet covered = new BitSet(data.numInstances());
        for(int i = 0; i < data.numInstances(); i++) {
            if(data.instance(i).value(att) <= val) {
                covered.set(i);
            }
        }
        return new BasicCachedRule(covered, val, "<=", att);
    }

    /**
     * Constructs a CachedRule that covers an instance iff that instance has
     * a values greater then or equal to another value in a specific column
     *
     * @param data Instances the rule will apply to
     * @param att Attribute to build the rule from
     * @param val double the instance's value must be greater then or equal to
     * @return CachedRule covering Instances with larger or equal values
     */
    public static BasicCachedRule GreaterOrEqualRule(Instances data, Attribute att, double val) {
        BitSet covered = new BitSet(data.numInstances());
        for(int i = 0; i < data.numInstances(); i++) {
            if(data.instance(i).value(att) >= val) {
                covered.set(i);
            }
        }
        return new BasicCachedRule(covered, val, ">=", att);
    }

    /** Instances this Rule Covers */
    private BitSet covered;

    /** A String representation of this */
    private String description;

    /**
     * Builds a BasicCachedRule from the String representation
     * and the instances it covers.
     *
     * @param covered BitSet with a bit set iff this covers that Instance from
     *                the same index in the dataset this rule was built for
     * @param description String representation
     */
    public BasicCachedRule(BitSet covered, String description) {
        this.covered = covered;
        this.description = description;
    }

    /**
     * Builds a BasicCachedRule from the instances it covers, the value and
     * attribute used, and a modifier String.
     *
     * @param covered BitSet with a bit set iff this covers that Instance from
     *                the same index in the dataset this rule was built for
     * @param val value used for comparison when determining what instances were
     *            covered
     * @param att Attribute used to build this rule
     * @param modifier String representation of how an instance's value relates
     *                 to the given value
     */
    public BasicCachedRule(BitSet covered, double val, String modifier, Attribute att) {
        this(covered, (att.isNominal() || att.isString() ?
                String.format("(%s %s %s)", att.name(), modifier, att.value((int) val)) :
                String.format("(%s %s %.3f)", att.name(), modifier, val)
        ));
    }

    @Override
    public String toString() {
        return description;
    }

    @Override
    public BitSet covered() {
        return covered;
    }
}