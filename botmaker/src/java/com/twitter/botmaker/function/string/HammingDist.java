package com.twitter.botmaker.function.string;

import java.math.BigInteger;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;

import com.twitter.botmaker.ASTNode;
import com.twitter.botmaker.Context;
import com.twitter.botmaker.compiler.ActionLevel;
import com.twitter.botmaker.compiler.BotMakerFunction;
import com.twitter.botmaker.compiler.exceptions.SemanticCheckFailure;
import com.twitter.botmaker.compiler.types.Type;
import com.twitter.botmaker.function.FunctionNode2;
import com.twitter.botmaker.runtime.Runtime;

@BotMakerFunction(
    name = {"HammingDist"},
    argTypes = {"String", "String"},
    arguments = {"hex string 1 for hamming distance", "hex string 2 for hamming distance"},
    returnType = "Long",
    description = "Calculate the hamming distance between two hex strings",
    actionLevel = ActionLevel.NO_ACTION,
    examples = {"HammingDist(\"ab\", \"ac\")"}
)
public class HammingDist extends FunctionNode2<Runtime, String, String> {
  private static final Signature SIGNATURE = new Signature(
      ImmutableList.of(Type.STRING, Type.STRING),
      Type.LONG
  );

  public HammingDist(String exprText, ImmutableList<ASTNode> children) throws SemanticCheckFailure {
    super(exprText, children);
  }

  @Override
  public Signature getSignature() {
    return SIGNATURE;
  }

  @Override
  protected CacheLevel getCacheLevel() {
    return CacheLevel.Global;
  }

  @Override
  protected Long apply(Context<Runtime> context, String string1, String string2) {
    return hammingDist(string1, string2);
  }

  static long hammingDist(BigInteger x, BigInteger y) throws ArithmeticException {
    long dist = 0L;
    BigInteger value = x.xor(y);
    while (value.compareTo(BigInteger.ZERO) >= 1) {
      value = value.and(value.subtract(BigInteger.ONE));
      dist = Math.addExact(dist, 1);
    }
    return dist;
  }

  static long hammingDist(String x, String y) throws ArithmeticException, NumberFormatException {
    Preconditions.checkArgument(x.length() == y.length(),
        "two strings must be of the same length");

    BigInteger num1 = new BigInteger('+' + x, 16);
    BigInteger num2 = new BigInteger('+' + y, 16);

    return hammingDist(num1, num2);
  }
}
