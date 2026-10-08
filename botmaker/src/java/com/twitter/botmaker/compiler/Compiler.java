package com.twitter.botmaker.compiler;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Collection;
import java.util.EnumSet;
import java.util.List;
import java.util.Map;
import javax.annotation.concurrent.NotThreadSafe;

import com.google.common.base.Optional;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.common.primitives.UnsignedBytes;

import org.antlr.runtime.tree.Tree;
import org.apache.commons.text.StringEscapeUtils;
import org.apache.thrift.TBase;

import com.twitter.botmaker.ASTNode;
import com.twitter.botmaker.antlr.BotMakerLexer;
import com.twitter.botmaker.compiler.exceptions.ParseFailure;
import com.twitter.botmaker.compiler.exceptions.SemanticCheckFailure;
import com.twitter.botmaker.compiler.tuples.Pair;
import com.twitter.botmaker.compiler.tuples.Tuple3;
import com.twitter.botmaker.compiler.types.Symbol;
import com.twitter.botmaker.compiler.types.ThriftStructType;
import com.twitter.botmaker.compiler.types.ThriftType;
import com.twitter.botmaker.compiler.types.Type;
import com.twitter.botmaker.compiler.types.TypeCompiler;
import com.twitter.botmaker.compiler.types.TypeContext;
import com.twitter.botmaker.function.Block;
import com.twitter.botmaker.function.Constant;
import com.twitter.botmaker.function.DerivedFeatureCall;
import com.twitter.botmaker.function.Do;
import com.twitter.botmaker.function.DoAsync;
import com.twitter.botmaker.function.DoSequential;
import com.twitter.botmaker.function.IfThenElse;
import com.twitter.botmaker.function.Val;
import com.twitter.botmaker.function.Variable;
import com.twitter.botmaker.function.With;
import com.twitter.botmaker.function.collection.Any;
import com.twitter.botmaker.function.collection.Contains;
import com.twitter.botmaker.function.collection.Copy;
import com.twitter.botmaker.function.collection.Filter;
import com.twitter.botmaker.function.collection.Flatten;
import com.twitter.botmaker.function.collection.Fold;
import com.twitter.botmaker.function.collection.ForEach;
import com.twitter.botmaker.function.collection.IndexAccessor;
import com.twitter.botmaker.function.collection.ListOperator;
import com.twitter.botmaker.function.collection.Sort;
import com.twitter.botmaker.function.collection.ToStructTuple;
import com.twitter.botmaker.function.collection.ToTuple;
import com.twitter.botmaker.function.conversion.Cast;
import com.twitter.botmaker.function.conversion.ThriftFromDict;
import com.twitter.botmaker.function.conversion.ToPair;
import com.twitter.botmaker.function.conversion.ToThrift;
import com.twitter.botmaker.function.feature.Exists;
import com.twitter.botmaker.function.feature.GetOrElse;
import com.twitter.botmaker.function.feature.InputFeature;
import com.twitter.botmaker.function.logic.Conjunction;
import com.twitter.botmaker.function.logic.Disjunction;
import com.twitter.botmaker.function.logic.Equals;
import com.twitter.botmaker.function.logic.GreaterThan;
import com.twitter.botmaker.function.logic.GreaterThanOrEquals;
import com.twitter.botmaker.function.logic.LessThan;
import com.twitter.botmaker.function.logic.LessThanOrEquals;
import com.twitter.botmaker.function.logic.NotEquals;
import com.twitter.botmaker.function.math.ArithmeticOperation;
import com.twitter.botmaker.function.math.Mod;
import com.twitter.botmaker.function.math.Negate;
import com.twitter.botmaker.function.misc.TestRun;
import com.twitter.botmaker.function.string.RegexMatch;
import com.twitter.botmaker.function.thrift.FieldAccessor;
import com.twitter.scrooge.ThriftStruct;

@NotThreadSafe
public class Compiler<E> {

  private static final int RADIX_16 = 16;

  private final TypeContext typeContext;

  public Compiler() {
    this(TypeContext.EMPTY);
  }

  public Compiler(TypeContext typeContext) {
    this.typeContext = typeContext;
  }

  public static <T> CompilerBuilder<T> builder(
      Class<? extends T> runtimeClass) throws SemanticCheckFailure {
    return new CompilerBuilder<T>(runtimeClass)
        .addPackage("com.twitter.botmaker.function");
  }

  public CompilerContext<E> createContext() {
    return new CompilerContext.Builder<E>()
        .setCompiler(this)
        .setTypeContext(typeContext)
        .setDerivedFeatures(DerivedFeatureProvider.EMPTY)
        .build();
  }

  public CompilerContext.Builder<E> createContextBuilder() {
    return new CompilerContext.Builder<E>()
        .setCompiler(this)
        .setTypeContext(typeContext);
  }

  public List<BotMakerDoc> getBotMakerDocs() {
    return BUILT_IN_DOCS;
  }

  public boolean containsFingerprint(Fingerprint fingerprint) {
    return fingerprint == Fingerprint.EMPTY;
  }

  protected Type getInputFeatureType(
      String namespace,
      String featureName) throws SemanticCheckFailure {
    return Type.OBJECT;
  }

  protected boolean containsFunction(String name) {
    return false;
  }

  protected ASTNode<E> createFunctionASTNode(
      CompilerContext<E> context,
      String funcName,
      ImmutableList<ASTNode> children) throws SemanticCheckFailure {
    throw new RuntimeException("Not implemented");
  }

  ASTNode createASTNodeTree(CompilerContext context, Tree root)
      throws ParseFailure, SemanticCheckFailure {

    ImmutableList<ASTNode> childASTNodes;
    ASTNode result = null;
    switch (root.getType()) {
      case BotMakerLexer.ARITHMETIC_OPERATION:
        assertNumChildren(root, 3);
        int arithmeticOperationNodeType = root.getChild(0).getType();
        childASTNodes = getChildrenASTNodesSkippingFirst(context, root);
        switch (arithmeticOperationNodeType) {
          case BotMakerLexer.SUM:
            result = ArithmeticOperation.ofSum(BuiltinFunctions.Sum.name(), childASTNodes);
            break;
          case BotMakerLexer.DIFFERENCE:
            result = ArithmeticOperation.ofDifference(
                BuiltinFunctions.Difference.name(), childASTNodes);
            break;
          case BotMakerLexer.PRODUCT:
            result = ArithmeticOperation.ofProduct(BuiltinFunctions.Product.name(), childASTNodes);
            break;
          case BotMakerLexer.QUOTIENT:
            result = ArithmeticOperation
                .ofQuotient(BuiltinFunctions.Quotient.name(), childASTNodes);
            break;
          case BotMakerLexer.MODULO:
            result = new Mod("Mod", childASTNodes);
            break;
          default:
        }
        break;
      case BotMakerLexer.STRATO_COLUMN_OP:
        String space;
        Tree path;
        String op;
        ArrayList<ASTNode> stratoOpAndArgs = new ArrayList<>();
        if (root.getChild(0).getType() == BotMakerLexer.FEATURE) {
          space = root.getChild(0).getText();
          path = root.getChild(1);
          op = root.getChild(2).getText();
          stratoOpAndArgs.add(Constant.of(op, op));
          for (int i = 3; i < root.getChildCount(); ++i) {
            stratoOpAndArgs.add(createASTNodeTree(context, root.getChild(i)));
          }
        } else {
          space = "";
          path = root.getChild(0);
          op = root.getChild(1).getText();
          stratoOpAndArgs.add(Constant.of(op, op));
          for (int i = 2; i < root.getChildCount(); ++i) {
            stratoOpAndArgs.add(createASTNodeTree(context, root.getChild(i)));
          }
        }
        StringBuilder pathStr = new StringBuilder();
        for (int i = 0; i < path.getChildCount(); ++i) {
          Tree pathComponent = path.getChild(i);
          StringBuilder componentStr = new StringBuilder();
          for (int j = 0; j < pathComponent.getChildCount(); ++j) {
            componentStr.append(j == 0 ? "" : "-");
            componentStr.append(pathComponent.getChild(j).getText());
          }
          pathStr.append(i == 0 ? "" : "/");
          pathStr.append(componentStr);
        }
        result = context.createFunctionASTNode(
            "DoStratoColumnOp",
            ImmutableList.of(
                Constant.of(space, space),
                Constant.of(pathStr.toString(), pathStr.toString()),
                Constant.of(op, op),
                context.createFunctionASTNode(
                    "ArgsToStratoJson",
                    ImmutableList.copyOf(stratoOpAndArgs)
                )
            )
        );
        break;
      case BotMakerLexer.FUNCTION:
        result = botmakerFunctionToASTNode(context, root);
        break;
      case BotMakerLexer.NULL:
        assertNumChildren(root, 0);
        result = Constant.NULL;
        break;
      case BotMakerLexer.BOOLEAN:
        assertNumChildren(root, 0);
        result = Constant.of(root.getText(), "TRUE".equals(root.getText()));
        break;
      case BotMakerLexer.STRING:
        assertNumChildren(root, 0);
        result = Constant.of(root.getText(), toStringConstant(root));
        break;
      case BotMakerLexer.SYMBOL:
        assertNumChildren(root, 0);
        result = Constant.of(root.getText(), toSymbolConstant(root));
        break;
      case BotMakerLexer.TEST_RUN:
        assertNumChildren(root, 2);
        result = toTestRun(context, BuiltinFunctions.TestRun.name(), root);
        break;
      case BotMakerLexer.SIGNED_LONG:
        assertNumChildren(root, 1);
        result = Constant.of(root.getChild(0).getText(),
            Long.parseLong(root.getChild(0).getText()) * (-1L));
        break;
      case BotMakerLexer.LONG:
        assertNumChildren(root, 0);
        result = Constant.of(root.getText(), Long.valueOf(root.getText()));
        break;
      case BotMakerLexer.SIGNED_DOUBLE:
        assertNumChildren(root, 1);
        result = Constant.of(root.getChild(0).getText(),
            Double.parseDouble(root.getChild(0).getText()) * (-1D));
        break;
      case BotMakerLexer.DOUBLE:
        assertNumChildren(root, 0);
        result = Constant.of(root.getText(), Double.valueOf(root.getText()));
        break;
      case BotMakerLexer.BINARY:
        assertNumChildren(root, 0);
        result = Constant.of(root.getText(), toBinaryConstant(root));
        break;
      case BotMakerLexer.CONJUNCTION:
        assertNumChildren(root, 2);
        result = new Conjunction(
            root.getText(),
            createASTNodeTree(context, root.getChild(0)),
            createASTNodeTree(context, root.getChild(1)));
        break;
      case BotMakerLexer.DISJUNCTION:
        assertNumChildren(root, 2);
        result = new Disjunction(
            root.getText(),
            createASTNodeTree(context, root.getChild(0)),
            createASTNodeTree(context, root.getChild(1)));
        break;
      case BotMakerLexer.PAIR:
        assertNumChildren(root, 2);
        result = toPair(context, root);
        break;
      case BotMakerLexer.VAL:
        result = toVal(context, root);
        break;
      case BotMakerLexer.WITH:
        result = toWith(context, root);
        break;
      case BotMakerLexer.VARIABLE:
        result = toVariable(context, root.getText());
        break;
      case BotMakerLexer.BLOCK:
        result = toBlock(context, root);
        break;
      case BotMakerLexer.FLAT_MAP:
        result = toFlatMap(context, root);
        break;
      case BotMakerLexer.FOREACH:
        result = toForEach(context, root);
        break;
      case BotMakerLexer.INDEX:
        result = toIndexAccessor(context, root);
        break;
      case BotMakerLexer.CAST:
        result = toCast(context, root);
        break;
      case BotMakerLexer.ACCESSOR:
        result = toFieldAccessor(context, root);
        break;
      case BotMakerLexer.IMPORT:
        result = toImport(context, root);
        break;
      case BotMakerLexer.IMPORT_AS:
        result = toImportAs(context, root);
        break;
      case BotMakerLexer.FEATURE:
        assertNumChildren(root, 0);
        String treeText = root.getText();
        Optional<DerivedFeature> parsedDerivedFeature = context.getDerivedFeature(treeText);
        if (parsedDerivedFeature.isPresent()) {
          result = toDerivedFeatureFunction(context, treeText, root.getLine(), ImmutableList.of());
        } else if (BuiltinFunctions.contains(treeText) || containsFunction(treeText)) {
          throw new SemanticCheckFailure(
              String.format("invalid input feature: %s is a function", treeText)
          );
        } else {
          Type type = getInputFeatureType(context.getInputFeatureNameSpace(), treeText);
          InputFeature inputFeature = InputFeature.of(treeText, type);
          context.trackUsage(inputFeature);
          result = inputFeature;
        }
        break;
      case BotMakerLexer.ARRAY:
        result = new ListOperator(root.getText(), getChildrenASTNodes(context, root));
        break;
      case BotMakerLexer.IF:
        result = new IfThenElse(root.getText(), getChildrenASTNodes(context, root));
        break;
      case BotMakerLexer.NEGATE:
        assertNumChildren(root, 1);
        result = new Negate(root.getText(), getChildrenASTNodes(context, root));
        break;

      case BotMakerLexer.EQUALS:
        assertNumChildren(root, 2);
        result = new Equals(root.getText(), getChildrenASTNodes(context, root));
        break;
      case BotMakerLexer.NOT_EQUALS:
        assertNumChildren(root, 2);
        result = new NotEquals(root.getText(), getChildrenASTNodes(context, root));
        break;
      case BotMakerLexer.LESS_THAN:
        assertNumChildren(root, 2);
        result = new LessThan(root.getText(), getChildrenASTNodes(context, root));
        break;
      case BotMakerLexer.LESS_THAN_OR_EQUALS:
        assertNumChildren(root, 2);
        result = new LessThanOrEquals(root.getText(), getChildrenASTNodes(context, root));
        break;
      case BotMakerLexer.GREATER_THAN:
        assertNumChildren(root, 2);
        result = new GreaterThan(root.getText(), getChildrenASTNodes(context, root));
        break;
      case BotMakerLexer.GREATER_THAN_OR_EQUALS:
        assertNumChildren(root, 2);
        result = new GreaterThanOrEquals(root.getText(), getChildrenASTNodes(context, root));
        break;
      case BotMakerLexer.MATCHES:
        assertNumChildren(root, 2);
        result = new RegexMatch(root.getText(), getChildrenASTNodes(context, root));
        break;
      default:
        throw new SemanticCheckFailure("not yet implemented semantics: " + root.getType());
    }

    return result;
  }

  static String toStringConstant(Tree tree) throws SemanticCheckFailure {
    String treeText = tree.getText();
    if (tree.getType() != BotMakerLexer.STRING) {
      throw new SemanticCheckFailure(String.format(
          "No string constant found for node with text %s", treeText));
    }
    int treeTextLength = treeText.length();
    if (treeTextLength < 2
        || treeText.charAt(0) != '"'
        || treeText.charAt(treeTextLength - 1) != '"') {
      throw new SemanticCheckFailure(String.format(
          "STRING node with invalid value: %s", treeText));
    }
    return StringEscapeUtils.unescapeJava(treeText.substring(1, treeTextLength - 1));
  }

  static Symbol toSymbolConstant(Tree tree) throws SemanticCheckFailure {
    String treeText = tree.getText();
    if (tree.getType() != BotMakerLexer.SYMBOL) {
      throw new SemanticCheckFailure(String.format(
          "No symbol constant found for node with text %s", treeText));
    }
    int treeTextLength = treeText.length();
    if (treeTextLength < 2
        || treeText.charAt(0) != '`'
        || treeText.charAt(treeTextLength - 1) != '`') {
      throw new SemanticCheckFailure(String.format(
          "SYMBOL node with invalid value: %s", treeText));
    }
    return new Symbol(StringEscapeUtils.unescapeJava(treeText.substring(1, treeTextLength - 1)));
  }

  static String getVariableName(Tree root) throws SemanticCheckFailure {
    String nodeText = root.getText();
    if (!nodeText.startsWith(":") || nodeText.length() <= 1) {
      throw new SemanticCheckFailure(String.format("Invalid variable name: %s", nodeText));
    }
    return nodeText;
  }

  static ByteBuffer toBinaryConstant(Tree tree) throws SemanticCheckFailure {
    String text = tree.getText();
    int length = text.length();
    if (length % 2 != 0 || length < 3 || !text.startsWith("0x")) {
      throw new SemanticCheckFailure(String.format(
          "No binary constant found for node with text %s", text));
    }

    byte[] arr = new byte[(length - 2) / 2];
    for (int i = 0, j = 2; i < arr.length; i++, j += 2) {
      arr[i] = UnsignedBytes.parseUnsignedByte(text.substring(j, j + 2), RADIX_16);
    }
    return ByteBuffer.wrap(arr);
  }

  static String getFeatureName(Tree tree) throws SemanticCheckFailure {
    switch (tree.getType()) {
      case BotMakerLexer.FEATURE:
        return tree.getText();
      case BotMakerLexer.STRING:
        return toStringConstant(tree);
      default:
        throw new SemanticCheckFailure(tree.getText() + " node is not FEATURE or STRING");
    }
  }

  public static String getFullName(Tree tree) throws SemanticCheckFailure {
    return getFullName(tree, tree.getChildCount());
  }

  static String getFullName(Tree tree, int count) throws SemanticCheckFailure {

    if (tree.getChildCount() < count) {
      throw new SemanticCheckFailure(tree.getText() + " node does not define a full name");
    }

    StringBuilder builder = new StringBuilder();
    builder.append(tree.getChild(0).getText());
    for (int index = 1; index < count; index++) {
      builder.append('.').append(tree.getChild(index).getText());
    }
    return builder.toString();
  }

  private static void assertNumChildren(Tree tree, int expectedNumChildren)
      throws SemanticCheckFailure {
    if (tree.getChildCount() != expectedNumChildren) {
      throw new SemanticCheckFailure(String.format(
          "%s node with invalid number of children: %d", tree.getText(), tree.getChildCount()));
    }
  }

  private static void assertNumChildren(Tree tree, int minimumNumChildren, int maximumNumChildren)
      throws SemanticCheckFailure {
    if (tree.getChildCount() < minimumNumChildren
        || tree.getChildCount() > maximumNumChildren) {
      throw new SemanticCheckFailure(String.format(
          "%s node with invalid number of children: %d", tree.getText(), tree.getChildCount()));
    }
  }

  private ASTNode botmakerFunctionToASTNode(CompilerContext context, Tree root)
      throws ParseFailure, SemanticCheckFailure {

    String funcName = getFullName(root.getChild(0));

    ASTNode result = null;
    ImmutableList<ASTNode> childASTNodes;



    if (BuiltinFunctions.contains(funcName)) {
      BuiltinFunctions funcNameEnum = BuiltinFunctions.forName(funcName);
      switch (funcNameEnum) {
        case All:
          assertNumChildren(root, 4);
          result = toAll(context, root);
          break;
        case Any:
          assertNumChildren(root, 4, 5);
          result = toAny(context, root);
          break;
        case Copy:
          assertNumChildren(root, 2, Integer.MAX_VALUE);
          result = toCopy(context, root);
          break;
        case Exists:
          assertNumChildren(root, 2);
          result = Exists.of(funcName, getFeatureName(root.getChild(1)));
          break;
        case Filter:
          assertNumChildren(root, 4);
          result = toFilter(context, funcName,
              root.getChild(1), root.getChild(2), root.getChild(3));
          break;
        case Foldl:
          assertNumChildren(root, 6);
          result = toFoldl(context, funcName, root);
          break;
        case Foldl1:
          assertNumChildren(root, 5);
          result = toFoldl1(context, funcName, root);
          break;
        case GetInputFeature:
          assertNumChildren(root, 2);
          result = toInputFeature(context, root);
          break;
        case Mock:
          result = toMock(context, funcName, root);
          break;
        case MockPass:
          result = toMockPass(context, funcName, root);
          break;
        case MockPassAll:
          result = toMockPassAll(context, funcName, root);
          break;
        case Sort:
          result = toSort(context, funcName, root);
          break;
        case StructTuple:
          result = toStructTuple(context, root);
          break;
        case TestRun:
          result = toTestRun(context, funcName, root);
          break;
        case ThriftFromDict:
          assertNumChildren(root, 3);
          result =
              thriftFromDict(
                  context,
                  funcName,
                  toStringConstant(root.getChild(1)),
                  root.getChild(2));
          break;
        case ToThrift:
          assertNumChildren(root, 3);
          result = toThrift(context, funcName,
              toStringConstant(root.getChild(1)), root.getChild(2));
          break;
        case Tuple:
          result = toTuple(context, root);
          break;
        default:
          break;
      }

      if (result == null) {
        childASTNodes = getChildrenASTNodesSkippingFirst(context, root);
        switch (funcNameEnum) {
          case Difference:
            result = ArithmeticOperation.ofDifference(funcName, childASTNodes);
            break;
          case Do:
            result = toDoOperations(context, funcName, childASTNodes, funcNameEnum);
            break;
          case DoAsync:
            result = toDoOperations(context, funcName, childASTNodes, funcNameEnum);
            break;
          case DoSequential:
            result = toDoOperations(context, funcName, childASTNodes, funcNameEnum);
            break;
          case GetOrElse:
            assertNumChildren(root, 3);
            final String featureName = getFeatureName(root.getChild(1));
            result = GetOrElse.of(funcName, featureName, childASTNodes);
            break;
          case Product:
            result = ArithmeticOperation.ofProduct(funcName, childASTNodes);
            break;
          case Quotient:
            result = ArithmeticOperation.ofQuotient(funcName, childASTNodes);
            break;
          case QuotientSafe:
            result = ArithmeticOperation.ofQuotientSafe(funcName, childASTNodes);
            break;
          case Sum:
            result = ArithmeticOperation.ofSum(funcName, childASTNodes);
            break;
          default:
            throw new SemanticCheckFailure(String.format("Can't find implementation for '%s'",
                funcName));
        }
      }
      return result;
    }

    Optional<DerivedFeature> df = context.getDerivedFeature(funcName);
    if (df.isPresent() && (!containsFunction(funcName) || df.get().isOverrideFunction())) {
      childASTNodes = getChildrenASTNodesSkippingFirst(context, root);
      try {
        result = toDerivedFeatureFunction(context, funcName, root.getLine(), childASTNodes);
      } catch (SemanticCheckFailure functionException) {
        try {
          result = context.createFunctionASTNode(funcName, childASTNodes);
        } catch (SemanticCheckFailure dfException) {
          throw functionException;
        }
      }
      return result;
    }

    if (context.containsFunction(funcName)) {
      childASTNodes = getChildrenASTNodesSkippingFirst(context, root);
      try {
        result = context.createFunctionASTNode(funcName, childASTNodes);
      } catch (Exception functionException) {
        try {
          result = toDerivedFeatureFunction(context, funcName, root.getLine(), childASTNodes);
        } catch (Exception dfException) {
          throw functionException;
        }
      }
      return result;
    }

    throw new SemanticCheckFailure("unrecognized function name: " + funcName);
  }

  private ASTNode toDoOperations(
      CompilerContext context,
      String exprText, ImmutableList<ASTNode> children,
      BuiltinFunctions funcNameEnum) throws SemanticCheckFailure {
    switch (funcNameEnum) {
      case Do:
        return new Do(exprText, children);
      case DoAsync:
        return new DoAsync(exprText, children);
      case DoSequential:
        return new DoSequential(exprText, children);
      default:
        throw new SemanticCheckFailure(
            "programming error: unknown Do* operation " + funcNameEnum);
    }
  }

  private ASTNode toIndexAccessor(CompilerContext context, Tree root)
      throws ParseFailure, SemanticCheckFailure {
    ASTNode exprNode = createASTNodeTree(context, root.getChild(0));
    ASTNode indexNode = createASTNodeTree(context, root.getChild(1));
    return IndexAccessor.mkIndexAccessor(root.getText(), exprNode, indexNode);
  }

  private ASTNode toCast(CompilerContext context, Tree root)
      throws ParseFailure, SemanticCheckFailure {
    Type type = TypeCompiler.getType(root.getChild(0), context.getTypeContext());
    ASTNode node = createASTNodeTree(context, root.getChild(1));
    return Cast.mkCast(root.getText(), type, node);
  }

  private ASTNode toFieldAccessor(CompilerContext context, Tree root)
      throws ParseFailure, SemanticCheckFailure {
    ASTNode node = createASTNodeTree(context, root.getChild(0));
    ImmutableList.Builder<String> fieldTokenListBuilder = ImmutableList.builder();
    for (int index = 1; index < root.getChildCount(); index++) {
      Tree fieldToken = root.getChild(index);
      fieldTokenListBuilder.add(fieldToken.getText());
    }
    return FieldAccessor.mkFieldAccessor(root.getText(), node, fieldTokenListBuilder.build());
  }

  private ASTNode thriftFromDict(
      CompilerContext context,
      String exprText,
      String className,
      Tree child
  ) throws SemanticCheckFailure {
    try {
      ASTNode node = createASTNodeTree(context, child);
      return ThriftFromDict.mkNewThriftObject(exprText, className, node);
    } catch (Exception e) {
      throw new SemanticCheckFailure(e);
    }
  }

  private ASTNode toThrift(
      CompilerContext context,
      String exprText,
      String className, Tree child) throws SemanticCheckFailure {
    try {
      ASTNode node = createASTNodeTree(context, child);
      return ToThrift.mkNewThriftObject(exprText, className, node);
    } catch (Exception e) {
      throw new SemanticCheckFailure(e);
    }
  }

  private ASTNode toImport(CompilerContext context, Tree root) throws SemanticCheckFailure {

    String clsName = getFullName(root);
    String alias = root.getChild(root.getChildCount() - 1).getText();
    Type type = TypeCompiler.getTypeFromClassName(clsName);
    if (type instanceof ThriftType) {
      context.defineType((ThriftType) type, alias);
    } else if (type instanceof ThriftStructType) {
      context.defineType((ThriftStructType) type, alias);
    } else {
      throw new SemanticCheckFailure(
          String.format(
              "class to be imported has to be either a TBase or ThriftStruct: %s.", clsName)
      );
    }

    return Constant.UNIT;
  }

  private ASTNode toImportAs(CompilerContext context, Tree root) throws SemanticCheckFailure {

    String cls = getFullName(root, root.getChildCount() - 1);
    String alias = root.getChild(root.getChildCount() - 1).getText();

    Class<?> clazz;
    try {
      clazz = ClassCache.forName(cls);
    } catch (Exception e) {
      throw new SemanticCheckFailure(
          String.format("cannot find class %s", cls)
      );
    }


    if (TBase.class.isAssignableFrom(clazz)) {
      ThriftType type = Type.thriftOf((Class<? extends TBase>) clazz);
      context.defineType(type, alias);
    } else if (ThriftStruct.class.isAssignableFrom(clazz)) {
      ThriftStructType type = Type.thriftStructOf((Class<? extends ThriftStruct>) clazz);
      context.defineType(type, alias);
    } else {
      throw new SemanticCheckFailure(
          String.format("class to be imported has to be either a TBase or ThriftStruct: %s.", cls)
      );
    }

    return Constant.UNIT;
  }

  private ImmutableList<ASTNode> getChildrenASTNodesSkippingFirst(
      CompilerContext context, Tree root) throws ParseFailure, SemanticCheckFailure {
    List<ASTNode> children = Lists.newArrayListWithCapacity(root.getChildCount() - 1);
    for (int i = 1; i < root.getChildCount(); i++) {
      children.add(createASTNodeTree(context, root.getChild(i)));
    }
    return ImmutableList.copyOf(children);
  }

  private ImmutableList<ASTNode> getChildrenASTNodes(
      CompilerContext context, Tree root) throws ParseFailure, SemanticCheckFailure {
    final ImmutableList.Builder<ASTNode> childrenBuilder = ImmutableList.builder();
    for (int i = 0; i < root.getChildCount(); i++) {
      childrenBuilder.add(createASTNodeTree(context, root.getChild(i)));
    }
    return childrenBuilder.build();
  }

  private ASTNode toAll(CompilerContext context, Tree root)
      throws ParseFailure, SemanticCheckFailure {

    return new Negate("Negate", ImmutableList.of(
        (ASTNode) new Contains("Contains", ImmutableList.of(
            toForEach(context, "FOREACH",
                root.getChild(1), root.getChild(3), root.getChild(2)),
            Constant.of("FALSE", false)
        ))
    ));
  }

  private ASTNode toAny(CompilerContext context, Tree root)
      throws ParseFailure, SemanticCheckFailure {

    Tree varNameTree = root.getChild(1);
    Tree conditionTree = root.getChild(2);
    Tree collectionTree = root.getChild(3);
    Optional<Tree> batchSizeTree = root.getChildCount() == 5
        ? Optional.of(root.getChild(4)) : Optional.absent();

    String varName = getVariableName(varNameTree);
    ASTNode collectionNode = createASTNodeTree(context, collectionTree);
    Optional<ASTNode> batchSizeNode = batchSizeTree.isPresent()
        ? Optional.of(createASTNodeTree(context, batchSizeTree.get()))
        : Optional.absent();

    final Type fromType;
    if (Collection.class.isAssignableFrom(collectionNode.getReturnType().typeBase)) {
      fromType = collectionNode.getReturnType().getTypeParams().get(0);
    } else {
      fromType = Type.OBJECT;
    }

    CompilerScope scope = context.addScope();

    Parameter conditionParameter = Parameter.scope(
        varName, fromType, context.currentScopeId(), collectionNode);
    scope.defineVariable(conditionParameter);

    ASTNode conditionNode = createASTNodeTree(context, conditionTree);
    context.removeScope();

    return Any.createAny("Any", varName, conditionNode,
        context.currentScopeId(), collectionNode, batchSizeNode);
  }

  private ASTNode toCopy(CompilerContext context, Tree root)
      throws ParseFailure, SemanticCheckFailure {
    ImmutableList<ASTNode> children = getChildrenASTNodesSkippingFirst(context, root);
    return Copy.of(children);
  }

  private ASTNode toFlatMap(CompilerContext context, Tree root)
      throws ParseFailure, SemanticCheckFailure {
    return toFlatMap(context,
        root.getText(), root.getChild(0), root.getChild(1), root.getChild(2));
  }

  private ASTNode toForEach(CompilerContext context, Tree root)
      throws ParseFailure, SemanticCheckFailure {
    return toForEach(context,
        root.getText(), root.getChild(0), root.getChild(1), root.getChild(2));
  }

  private ASTNode toFlatMap(
      CompilerContext context,
      String exprText, Tree variableNameTree,
      Tree collectionTree, Tree bodyTree) throws ParseFailure, SemanticCheckFailure {
    String variableName = getVariableName(variableNameTree);
    ASTNode collectionNode = createASTNodeTree(context, collectionTree);

    Type variableType = Type.OBJECT;
    if (Collection.class.isAssignableFrom(collectionNode.getReturnType().typeBase)) {
      variableType = collectionNode.getReturnType().getTypeParams().get(0);
    }

    CompilerScope scope = context.addScope();
    Parameter parameter = Parameter.scope(
        variableName, variableType, context.currentScopeId(), collectionNode);
    scope.defineVariable(parameter);

    ASTNode bodyNode = createASTNodeTree(context, bodyTree);
    context.removeScope();

    ASTNode forEachNode = ForEach.of(
        exprText, variableName, context.currentScopeId(), collectionNode, bodyNode);
    return new Flatten(exprText, ImmutableList.of(forEachNode));
  }

  private ASTNode toForEach(
      CompilerContext context,
      String exprText, Tree variableNameTree,
      Tree collectionTree, Tree bodyTree) throws ParseFailure, SemanticCheckFailure {
    String variableName = getVariableName(variableNameTree);
    ASTNode collectionNode = createASTNodeTree(context, collectionTree);

    Type variableType = Type.OBJECT;
    if (Collection.class.isAssignableFrom(collectionNode.getReturnType().typeBase)) {
      variableType = collectionNode.getReturnType().getTypeParams().get(0);
    }

    CompilerScope scope = context.addScope();
    Parameter parameter = Parameter.scope(
        variableName, variableType, context.currentScopeId(), collectionNode);
    scope.defineVariable(parameter);

    ASTNode bodyNode = createASTNodeTree(context, bodyTree);
    context.removeScope();

    ASTNode result = ForEach.of(
        exprText, variableName, context.currentScopeId(), collectionNode, bodyNode);
    return result;
  }

  private ASTNode toFilter(
      CompilerContext context,
      String exprText, Tree variableNameTree, Tree conditionTree, Tree collectionTree)
      throws ParseFailure, SemanticCheckFailure {
    String variableName = getVariableName(variableNameTree);
    ASTNode collectionNode = createASTNodeTree(context, collectionTree);
    Type variableType = Type.OBJECT;
    if (Collection.class.isAssignableFrom(collectionNode.getReturnType().typeBase)) {
      variableType = collectionNode.getReturnType().getTypeParams().get(0);
    }

    CompilerScope scope = context.addScope();
    Parameter parameter = Parameter.scope(
        variableName, variableType, context.currentScopeId(), collectionNode);
    scope.defineVariable(parameter);

    ASTNode conditionNode = createASTNodeTree(context, conditionTree);
    context.removeScope();

    ASTNode result = Filter.of(
        exprText, variableName, conditionNode, context.currentScopeId(), collectionNode);
    return result;
  }

  private ASTNode toFoldl(CompilerContext context, String exprText, Tree root)
      throws ParseFailure, SemanticCheckFailure {
    String varA = getVariableName(root.getChild(1));
    String varB = getVariableName(root.getChild(2));
    Tree funcTree = root.getChild(3);
    ASTNode seedValueNode = createASTNodeTree(context, root.getChild(4));
    ASTNode collectionNode = createASTNodeTree(context, root.getChild(5));
    Type fromType = Type.OBJECT;
    if (Collection.class.isAssignableFrom(collectionNode.getReturnType().typeBase)) {
      fromType = collectionNode.getReturnType().getTypeParams().get(0);
    }
    Type seedType = seedValueNode.getReturnType();

    CompilerScope scope = context.addScope();
    long scopeId = context.currentScopeId();
    Parameter parameterA = Parameter.scope(
        varA, seedType, scopeId, seedValueNode);
    scope.defineVariable(parameterA);

    Parameter parameterB = Parameter.scope(
        varB, fromType, scopeId, collectionNode);
    scope.defineVariable(parameterB);

    ASTNode funcNode = createASTNodeTree(context, funcTree);
    if (Type.isDivergentTo(funcNode.getReturnType(), seedType)) {
      throw new SemanticCheckFailure(
          String.format(
              "function passed to Foldl() returns %s but seed value is a %s",
              funcNode.getReturnType(),
              seedType));
    }

    context.removeScope();

    ASTNode result = Fold.createFoldl(exprText, varA, varB, funcNode, seedValueNode,
        context.currentScopeId(), collectionNode);
    return result;
  }

  private ASTNode toFoldl1(CompilerContext context, String exprText, Tree root)
      throws ParseFailure, SemanticCheckFailure {
    String varA = getVariableName(root.getChild(1));
    String varB = getVariableName(root.getChild(2));
    ASTNode collectionNode = createASTNodeTree(context, root.getChild(4));
    Type variableType = Type.OBJECT;
    if (Collection.class.isAssignableFrom(collectionNode.getReturnType().typeBase)) {
      variableType = collectionNode.getReturnType().getTypeParams().get(0);
    }

    CompilerScope scope = context.addScope();
    long scopeId = context.currentScopeId();

    Parameter parameterA = Parameter.scope(
        varA, variableType, scopeId, collectionNode);
    scope.defineVariable(parameterA);

    Parameter parameterB = Parameter.scope(
        varB, variableType, scopeId, collectionNode);
    scope.defineVariable(parameterB);

    ASTNode funcNode = createASTNodeTree(context, root.getChild(3));
    if (Type.isDivergentTo(funcNode.getReturnType(), variableType)) {
      throw new SemanticCheckFailure(
          String.format(
              "Foldl1 is expected to return %s but passed function returns %s",
              variableType,
              funcNode.getReturnType()));
    }

    context.removeScope();

    ASTNode result = Fold.createFoldl1(
        exprText, varA, varB, funcNode, context.currentScopeId(), collectionNode);
    return result;
  }

  private ASTNode toInputFeature(CompilerContext context, Tree root)
      throws SemanticCheckFailure {
    String featureName = toStringConstant(root.getChild(1));
    Type type = getInputFeatureType(context.getInputFeatureNameSpace(), featureName);

    InputFeature result = InputFeature.of(featureName, type);
    context.trackUsage(result);
    return result;
  }

  private ASTNode toSort(CompilerContext context, String exprText, Tree root)
      throws ParseFailure, SemanticCheckFailure {

    if (root.getChildCount() == 5) {
      String varA = getVariableName(root.getChild(1));
      String varB = getVariableName(root.getChild(2));
      ASTNode collectionNode = createASTNodeTree(context, root.getChild(4));
      Type variableType = Type.OBJECT;
      if (Collection.class.isAssignableFrom(collectionNode.getReturnType().typeBase)) {
        variableType = collectionNode.getReturnType().getTypeParams().get(0);
      }

      CompilerScope scope = context.addScope();
      long scopeId = context.currentScopeId();

      Parameter parameterA = Parameter.scope(
          varA, variableType, scopeId, collectionNode);
      scope.defineVariable(parameterA);

      Parameter parameterB = Parameter.scope(
          varB, variableType, scopeId, collectionNode);
      scope.defineVariable(parameterB);

      ASTNode funcNode = createASTNodeTree(context, root.getChild(3));
      if (Type.isDivergentTo(funcNode.getReturnType(), Type.LONG)) {
        throw new SemanticCheckFailure(
            String.format(
                "Sort is expected to return %s but passed function returns %s",
                Type.LONG,
                funcNode.getReturnType()));
      }

      context.removeScope();

      ASTNode result = Sort.toSort(
          exprText, varA, varB, funcNode, context.currentScopeId(), collectionNode);
      return result;
    } else if (root.getChildCount() == 2) {
      ASTNode collectionNode = createASTNodeTree(context, root.getChild(1));
      Type elementType = Type.OBJECT;
      if (Collection.class.isAssignableFrom(collectionNode.getReturnType().typeBase)) {
        elementType = collectionNode.getReturnType().getTypeParams().get(0);
      }

      if (!Comparable.class.isAssignableFrom(elementType.typeBase)) {
        throw new SemanticCheckFailure(
            String.format("Sort expects a collection of %s objects, but gets %s",
                Comparable.class.getName(),
                elementType)
        );
      }

      ASTNode result = Sort.toSort(exprText, collectionNode);
      return result;
    } else {
      throw new SemanticCheckFailure(String.format(
          "%s node with invalid number of children: %d", root.getText(), root.getChildCount()));
    }
  }

  private ASTNode toBlock(CompilerContext context, Tree root)
      throws ParseFailure, SemanticCheckFailure {
    if (root.getChildCount() == 0) {
      throw new SemanticCheckFailure("block body cannot be empty");
    }

    CompilerScope currentScope = context.addScope();
    ImmutableList.Builder<ASTNode> childrenBuilder = ImmutableList.builder();
    for (int i = 0; i < root.getChildCount(); i++) {
      childrenBuilder.add(createASTNodeTree(context, root.getChild(i)));
    }
    ImmutableList.Builder<Pair<String, Type>> variableListBuilder = ImmutableList.builder();
    for (Map.Entry<String, Parameter> entry : currentScope.getVariables().entrySet()) {
      variableListBuilder.add(Pair.of(entry.getKey(), entry.getValue().getReturnType()));
    }
    context.removeScope();
    return new Block(
        root.getText(),
        childrenBuilder.build(),
        variableListBuilder.build()
    );
  }

  private Variable toVariable(
      CompilerContext context, String variableText) throws SemanticCheckFailure {
    return context.getVariable(variableText);
  }

  private ASTNode toPair(CompilerContext context, Tree root)
      throws ParseFailure, SemanticCheckFailure {
    ASTNode<E> firstNode = createASTNodeTree(context, root.getChild(0));
    ASTNode<E> secondNode = createASTNodeTree(context, root.getChild(1));
    return ToPair.of(root.getText(), firstNode, secondNode);
  }

  private ASTNode toTuple(CompilerContext context, Tree root)
      throws ParseFailure, SemanticCheckFailure {
    List<ASTNode> children = Lists.newArrayListWithCapacity(root.getChildCount() - 1);
    for (int index = 1; index < root.getChildCount(); index++) {
      ASTNode child = createASTNodeTree(context, root.getChild(index));
      children.add(child);
    }
    return ToTuple.of(children);
  }

  private ASTNode toStructTuple(CompilerContext context, Tree root)
      throws ParseFailure, SemanticCheckFailure {
    List<ASTNode> children = Lists.newArrayListWithCapacity(root.getChildCount() - 1);
    for (int index = 1; index < root.getChildCount(); index++) {
      ASTNode child = createASTNodeTree(context, root.getChild(index));
      children.add(child);
    }
    return ToStructTuple.of("StructTuple", ImmutableList.copyOf(children));
  }

  private ASTNode toVal(CompilerContext context, Tree root)
      throws ParseFailure, SemanticCheckFailure {
    Pair<String, ASTNode> assignment = toAssignment(context, root.getChild(0));
    context.defineVariable(assignment.getFirst(), assignment.getSecond());
    return new Val(root.getText(), assignment);
  }

  private ASTNode toWith(CompilerContext context, Tree root)
      throws ParseFailure, SemanticCheckFailure {
    final int variableCount = root.getChildCount() - 1;
    context.addScope();
    final List<Pair<String, ASTNode>> variables = Lists.newArrayListWithCapacity(variableCount);
    for (int i = 0; i < variableCount; i++) {
      Pair<String, ASTNode> assignment = toAssignment(context, root.getChild(i));
      variables.add(assignment);
      context.defineVariable(assignment.getFirst(), assignment.getSecond());
    }

    final ASTNode bodyASTNode = createASTNodeTree(context, root.getChild(variableCount));
    context.removeScope();

    final ASTNode result = With.of(root.getText(), variables, bodyASTNode);
    return result;
  }

  private Pair<String, ASTNode> toAssignment(CompilerContext context, Tree root)
      throws ParseFailure, SemanticCheckFailure {
    final String variableName = getVariableName(root.getChild(0));
    final Tree variableRhsNode = root.getChild(1);
    final ASTNode varExprASTNode = createASTNodeTree(context, variableRhsNode);

    if (varExprASTNode.getReturnType() == Type.SYMBOL) {
      throw new SemanticCheckFailure("Cannot assign SYMBOL type to variable");
    }

    return Pair.of(variableName, varExprASTNode);
  }

  private ASTNode toDerivedFeatureFunction(
      CompilerContext context, String funcName, int lineNo, List<ASTNode> children
  ) throws ParseFailure, SemanticCheckFailure {
    if (context.containsMacro(funcName)) {
      throw new SemanticCheckFailure(String.format(
          "Recursive invocation of derived feature function %s is prohibited", funcName));
    }

    Optional<DerivedFeature> dfOpt = context.getDerivedFeature(funcName);
    if (!dfOpt.isPresent()) {
      throw new SemanticCheckFailure(String.format(
          "%s is not a derived feature function", funcName));
    }

    DerivedFeature derivedFeature = dfOpt.get();
    context.trackUsage(derivedFeature);

    List<Tuple3<Tree, String, Optional<Tree>>> params = derivedFeature.getParams();
    boolean isLogged = derivedFeature.isLogged();

    List<ASTNode> args = children;
    if (params.size() != children.size()) {

      if (params.size() < children.size()
          || !params.get(children.size()).getThird().isPresent()) {
        throw new SemanticCheckFailure(String.format(
            "Derived Feature %s has %d parameters, received %d",
            funcName, params.size(), children.size()));
      }

      ImmutableList.Builder<ASTNode> builder = ImmutableList.builder();
      builder.addAll(children);
      for (int i = children.size(); i < params.size(); i++) {
        Tree tree = params.get(i).getThird().get();
        ASTNode child = createASTNodeTree(context, tree);
        builder.add(child);
      }
      args = builder.build();
    }

    List<Pair<String, ASTNode>> variables = Lists.newArrayListWithCapacity(params.size());
    context.addScope();
    for (int i = 0; i < params.size(); i++) {
      String paramName = params.get(i).getSecond();
      Type paramTp =
          TypeCompiler.getType(params.get(i).getFirst(), context.getTypeContext());
      ASTNode arg = args.get(i);
      if (Type.isDivergentTo(paramTp, arg.getReturnType())) {
        throw new SemanticCheckFailure(String.format(
            "Parameter #%d of %s should be %s, received %s instead",
            i + 1, funcName, paramTp, arg.getReturnType()));
      }
      variables.add(Pair.of(paramName, arg));

      context.defineVariable(paramName, paramTp, arg);
    }

    context.addMacro(funcName);
    ASTNode bodyASTNode = context.getCachedDFBody(derivedFeature);
    context.removeMacro(funcName);
    context.removeScope();

    return DerivedFeatureCall.of(derivedFeature, isLogged, lineNo, variables, bodyASTNode);
  }

  private ASTNode toMock(
      CompilerContext context,
      String exprText,
      Tree root) throws SemanticCheckFailure {
    try {
      if (root.getChildCount() == 3) {
        String funcName = toStringConstant(root.getChild(1));
        ASTNode bodyNode = createASTNodeTree(context, root.getChild(2));
        context.addMock(funcName, bodyNode);
      } else if (root.getChildCount() == 4) {
        for (int i = 1; i < root.getChildCount() - 1; i++) {
          String funcName = toStringConstant(root.getChild(1));
          String variableName = getVariableName(root.getChild(2));
          Type variableType = Type.listOf(Type.OBJECT);

          CompilerScope scope = context.addScope();
          Parameter parameter = Parameter.open(variableName, variableType);
          scope.defineVariable(parameter);

          ASTNode bodyNode = createASTNodeTree(context, root.getChild(3));
          context.removeScope();

          context.addMock(funcName, variableName, bodyNode);
        }
      } else {
        throw new SemanticCheckFailure(
            String.format("Mock function expects either 1 or 3 arguments but %d received",
                root.getChildCount() - 1)
        );
      }
    } catch (Exception e) {
      throw new SemanticCheckFailure(e);
    }

    return Constant.UNIT;
  }

  private ASTNode toMockPass(
      CompilerContext context,
      String exprText,
      Tree root) throws SemanticCheckFailure {
    try {
      if (root.getChildCount() == 2) {
        String funcName = toStringConstant(root.getChild(1));
        context.addMockPass(funcName);
      } else {
        throw new SemanticCheckFailure(
            String.format("MockPass function expects either 1 arguments but %d received",
                root.getChildCount() - 1)
        );
      }
    } catch (Exception e) {
      throw new SemanticCheckFailure(e);
    }

    return Constant.UNIT;
  }

  private ASTNode toMockPassAll(
      CompilerContext context,
      String exprText,
      Tree root) throws SemanticCheckFailure {
    try {
      if (root.getChildCount() == 2) {
        String packageName = toStringConstant(root.getChild(1));
        context.addMockPackage(packageName);
      } else {
        throw new SemanticCheckFailure(
            String.format("MockPassAll function expects 1 arguments but %d received",
                root.getChildCount() - 1)
        );
      }
    } catch (Exception e) {
      throw new SemanticCheckFailure(e);
    }

    return Constant.UNIT;
  }

  private ASTNode toTestRun(
      CompilerContext context,
      String exprText,
      Tree root) throws SemanticCheckFailure {
    try {

      if (root.getChildCount() != 2) {
        throw new SemanticCheckFailure(
            String.format("TestRun function expects 1 arguments but %d received",
                root.getChildCount() - 1)
        );
      }

      context.addScope();
      context.addTestScope();
      Tree target = root.getChild(1);
      ASTNode node = createASTNodeTree(context, target);
      ASTNode testRun = new TestRun("TestRun", ImmutableList.of(node));

      context.removeTestScope();
      context.removeScope();
      return testRun;
    } catch (Exception e) {
      throw new SemanticCheckFailure(e);
    }
  }

  public enum BuiltinFunctions {
    All,
    Any,
    Copy,
    Difference,
    Do,
    DoAsync,
    DoSequential,
    Exists,
    Filter,
    Foldl,
    Foldl1,
    GetInputFeature,
    GetOrElse,
    Mock,
    MockPass,
    MockPassAll,
    Product,
    Quotient,
    QuotientSafe,
    Sort,
    StructTuple,
    Sum,
    Symbols,
    TestRun,
    ThriftFromDict,
    ToThrift,
    Tuple;

    public static final Map<String, BuiltinFunctions> BY_NAME =
        Maps.uniqueIndex(EnumSet.allOf(BuiltinFunctions.class), item -> item.name());

    public static BuiltinFunctions forName(String name) {
      final BuiltinFunctions enumValue = BY_NAME.get(name);
      return enumValue;
    }

    public static boolean contains(String name) {
      return BY_NAME.containsKey(name);
    }
  }

  private static final ImmutableList<BotMakerDoc> BUILT_IN_DOCS = ImmutableList.of(
      BotMakerDoc.of(
          "All",
          "Checks whether all of the elements in the collection satisfy the boolean condition."
              + "For example, All(:v, :v>0, Set(1,2)), returns true.",
          ActionLevel.NO_ACTION,
          new String[]{"Variable", "Boolean", "Collection"},
          new String[]{"the variable name", "the boolean condition", "the collection"},
          "Boolean"
      ),
      BotMakerDoc.of(
          "Any",
          "Checks whether any of the elements in the collection satisfy the boolean condition."
              + "For example, Any(:v, :v>1, Set(1,2)), returns true. The boolean condition passed "
              + "in must be idempotent, where there are no side effects if it is called more than "
              + "once or not at all. By default Any will test the condition against smaller "
              + "batches from the collection, and the batch size can be specified with an "
              + "optional fourth argument.",
          ActionLevel.NO_ACTION,
          new String[]{"Variable", "Boolean", "Collection", "Long (optional)"},
          new String[]{"the variable name", "the boolean condition", "the collection",
              "batch size (optional)"},
          "Boolean"
      ),
      BotMakerDoc.of(
          "Copy",
          "Copy a Map or StructTuple and optionally provide new values.",
          ActionLevel.NO_ACTION,
          new String[]{
              "Map or StructTuple",
              "Pair..."},
          new String[]{
              "The Map or StructTuple to be copied.",
              "Key value pairs to add in the new Map or StructTuple."},
          "Map or StructTuple"
      ),
      BotMakerDoc.of(
          "Difference",
          "Calculate the difference between two numbers",
          ActionLevel.NO_ACTION,
          new String[]{"Number", "Number"},
          new String[]{"the first number", "the second number"},
          "Number"
      ),
      BotMakerDoc.of(
          "Do",
          "Executes all given expressions and returns the union of their results.",
          ActionLevel.NO_ACTION,
          new String[]{"T..."},
          new String[]{"the expressions"},
          "Set<T>"
      ),
      BotMakerDoc.of(
          "DoAsync",
          "Executes all given expressions asynchronously and returns a UNIT.",
          ActionLevel.NO_ACTION,
          new String[]{"T..."},
          new String[]{"the expressions"},
          "UNIT"
      ),
      BotMakerDoc.of(
          "DoSequential",
          "Executes all given expressions sequentially "
              + "and returns the union of their results",
          ActionLevel.NO_ACTION,
          new String[]{"T..."},
          new String[]{"the expressions"},
          "Set<T>"
      ),
      BotMakerDoc.of(
          "Exists",
          "Whether the given feature exists among the input features.",
          ActionLevel.NO_ACTION,
          new String[]{"FeatureName"},
          new String[]{"the feature name"},
          "Boolean"
      ),
      BotMakerDoc.of(
          "Filter",
          "Returns the subset of elements in the list that satisfy the condition. "
              + "For example, Filter(:v, Mod(:v, 3) == 1, List(1,2,3,4,5)), returns [1, 4].",
          ActionLevel.NO_ACTION,
          new String[]{"Variable", "Boolean", "List"},
          new String[]{
              "the variable name",
              "the boolean expression",
              "the list"
          },
          "List"
      ),
      BotMakerDoc.of(
          "Foldl",
          "Higher order foldl function. For example,"
              + "Foldl(:accum, :elem, ListConcat(List(:elem), :accum), List(),"
              + "List(1,2,998,42,64,5)), reverses the list, returning [5, 64, 42, 998, 2, 1].",
          ActionLevel.NO_ACTION,
          new String[]{"Variable", "Variable", "T", "T", "Collection"},
          new String[]{
              "the accumulator variable name",
              "the element variable name",
              "the operation to run",
              "initial value for accumulator",
              "the collection"
          },
          "T"
      ),
      BotMakerDoc.of(
          "Foldl1",
          "Higher order foldl1 function. For example,"
              + "Foldl1(:a, :b, Sum(:a, :b), List(1,2,3,4,5)), adds all the elements together,"
              + "which returns 15.",
          ActionLevel.NO_ACTION,
          new String[]{"Variable", "Variable", "T", "Collection"},
          new String[]{
              "the variable 1 name",
              "the variable 2 name",
              "the operation to run",
              "the collection"
          },
          "T"
      ),
      BotMakerDoc.of(
          "GetInputFeature",
          "Get input feature value by name. For example, "
              + "GetInputFeature(\"profile.auth.hashedPassword.before\")",
          ActionLevel.NO_ACTION,
          new String[]{"FeatureName"},
          new String[]{"input feature name"},
          "Object"
      ),
      BotMakerDoc.of(
          "GetOrElse",
          "Get the value of the feature name or the value supplied if not."
              + "NOTE: The type of the feature has to match the type else value. Things will"
              + "fail in runtime otherwise.",
          ActionLevel.NO_ACTION,
          new String[]{"FeatureName", "T"},
          new String[]{
              "feature name",
              "else value"
          },
          "T"
      ),
      BotMakerDoc.of(
          "Mock",
          "Mock a function for test. You can either specify a constant value for the "
              + "return value or intercept the arguments with a mock expression. For example, "
              + "Mock( \"GetUserName\", \"Ted Bot\");"
              + "Mock( \"GetUserName\", :args, { ToString(:args[0]) } );",
          ActionLevel.NO_ACTION,
          new String[]{"String", "..."},
          new String[]{"Function name", "mock value"},
          "UNIT"
      ),
      BotMakerDoc.of(
          "MockPass",
          "Do not mock a function and use the real ASTNode instead. For example, "
              + "MockPass( \"GetUserName\");",
          ActionLevel.NO_ACTION,
          new String[]{"String"},
          new String[]{"Function name"},
          "UNIT"
      ),
      BotMakerDoc.of(
          "MockPassAll",
          "Do not mock all functions under the given package hierarchy. For example, "
              + "MockPassAll( \"com.twitter.myapp.function\");",
          ActionLevel.NO_ACTION,
          new String[]{"String"},
          new String[]{"Package name"},
          "UNIT"
      ),
      BotMakerDoc.of(
          "Product",
          "Calculate the product of two numbers",
          ActionLevel.NO_ACTION,
          new String[]{"Number", "Number"},
          new String[]{"the first number", "the second number"},
          "Number"
      ),
      BotMakerDoc.of(
          "Quotient",
          "Calculate the quotient of two numbers",
          ActionLevel.NO_ACTION,
          new String[]{"Number", "Number"},
          new String[]{"the first number", "the second number"},
          "Number"
      ),
      BotMakerDoc.of(
          "QuotientSafe",
          "The quotient of the given numbers. Returns 0 if divisor is 0.",
          ActionLevel.NO_ACTION,
          new String[]{"Number", "Number"},
          new String[]{"the first number", "the second number"},
          "Number"
      ),
      BotMakerDoc.of(
          "Sort",
          "Sort a collection. This function has two forms. For example, "
              + "Sort([3,2,1]);"
              + "Sort(:a, :b, If(:a > :b, 1, If( :a == :b, 0, -1)), [3,2,1]);",
          ActionLevel.NO_ACTION,
          new String[]{"..."},
          new String[]{},
          "Collection"
      ),
      BotMakerDoc.of(
          "StructTuple",
          "Create a StructTuple object."
              + "val :user = StructTuple( \"id\" : 123, \"name\": \"abc\");"
              + ":user.id == 123;"
              + ":user.name == \"abc\";",
          ActionLevel.NO_ACTION,
          new String[]{"..."},
          new String[]{},
          "StructTuple"
      ),
      BotMakerDoc.of(
          "Sum",
          "Given two numbers, calculate the sum of two numbers;"
              + " given two sub-strings, returns the concatenated string",
          ActionLevel.NO_ACTION,
          new String[]{"Number/String", "Number/String"},
          new String[]{"the first number or string", "the second number or string"},
          "Number/String"
      ),
      BotMakerDoc.of(
          "ThriftFromDict",
          "Construct a thrift object based on the given map",
          ActionLevel.NO_ACTION,
          new String[]{"String", "Map<String, Object>"},
          new String[]{"thrift class name", "dict"},
          "Thrift"
      ),
      BotMakerDoc.of(
          "ToThrift",
          "Cast an object to a thrift type",
          ActionLevel.NO_ACTION,
          new String[]{"Object"},
          new String[]{"the element to be cast"},
          "Thrift"
      ),
      BotMakerDoc.of(
          "Tuple",
          "Make a tuple.",
          ActionLevel.NO_ACTION,
          new String[]{"T..."},
          new String[]{"elements in the tuple"},
          "Tuple"
      ),
      BotMakerDoc.of(
          "ToTuple",
          "Make a tuple.",
          ActionLevel.NO_ACTION,
          new String[]{"T..."},
          new String[]{"elements in the tuple"},
          "Tuple"
      )
  );
}
